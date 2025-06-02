import tensorflow as tf
import numpy as np

np.random.seed(0)

from model.generate_anchor import generate_anchors
from model.bbox_transform import clip_boxes
from model.ellipse_transform import ellipse_transform_inv, ellipse2box
#from nms.cpu_nms import cpu_nms
#from nms.gpu_nms import gpu_nms
from nms.nms import nms

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = tf.reshape(tf.where((ws >= min_size) & (hs >= min_size)),[-1])

    return keep


class EllipseProposalLayer(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(EllipseProposalLayer, self).__init__()
        self._cfg = dict(cfg)
        self._preprocess()

    def _preprocess(self): 
        # pre-computing stuff for making anchor later
        self._im_info = (self._cfg['MAX_SIZE'], self._cfg['MAX_SIZE'])
        base_anchors = generate_anchors(
            base_size=self._cfg['RPN_FEAT_STRIDE'],
            ratios=[1],
            scales=np.array(self._cfg['ANCHOR_SCALES'], dtype=np.float32))
        num_anchors = base_anchors.shape[0]

        ### print(num_anchors)
        feat_stride = self._cfg['RPN_FEAT_STRIDE']
        feat_width = self._cfg['MAX_SIZE'] // self._cfg['RPN_FEAT_STRIDE']
        feat_height = feat_width

        shift_x = np.arange(0, feat_width) * feat_stride
        shift_y = np.arange(0, feat_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors
        K = shifts.shape[0]
        
        base_anchors_reshaped = tf.reshape(base_anchors, [1, A, 4])
        shifts_reshaped = tf.reshape(shifts, [1, K, 4])
        shifts_transposed = tf.transpose(shifts_reshaped, perm=[1, 0, 2])  # shape (K, 1, 4)

        anchors = tf.cast(base_anchors_reshaped, tf.float32) + tf.cast(shifts_transposed, tf.float32)  # broadcast addition -> shape (K, A, 4)

        anchors = tf.reshape(anchors, [K * A, 4])

        self._feat_height = feat_height
        self._feat_width = feat_width
        self._anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

    
    def call(self, out_cls, out_ellipse):
        """
        out_cls: (feat_height, feat_width, anchors, 2) FloatVariable
        out_ellipse: (feat_height, feat_width, anchors, 5) FloatVariable
        """
        
        scores = tf.nn.softmax(out_cls, axis=3)[..., 1]
        scores = tf.stop_gradient(scores)
        scores = tf.reshape(scores, [-1, 1])
        
        out_ellipse = tf.stop_gradient(out_ellipse)
        ellipse_deltas = tf.reshape(out_ellipse, [-1, 5])

        # 1. Generate proposals from ellipse deltas and shifted anchors
        # Convert anchors into proposals via ellipse transformations
        # Convert ellipse into bbox proposals
        ellipses = ellipse_transform_inv(self._anchors, ellipse_deltas)
        boxes = ellipse2box(ellipses, self._cfg['ELLIPSE_PAD'])

        # 2. clip predicted boxes to image
        boxes = clip_boxes(boxes, self._im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTICE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(boxes, self._cfg['TEST.RPN_MIN_SIZE'])
        
        boxes = tf.gather(boxes, keep,axis=0)
        ellipses = tf.gather(ellipses, keep,axis=0)
        scores = tf.gather(scores, keep)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = tf.argsort(tf.reshape(scores, [-1]), direction='DESCENDING')
        if self._cfg['TEST.RPN_PRE_NMS_TOP_N'] > 0:
            order = order[:self._cfg['TEST.RPN_PRE_NMS_TOP_N']]
        
        boxes = tf.gather(boxes, order,axis=0)
        ellipses = tf.gather(ellipses, order,axis=0)
        scores = tf.gather(scores, order)

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        #if self._cfg['USE_GPU_NMS']:
        #    nms = gpu_nms
        #else:
        #    nms = cpu_nms
        boxes_np = boxes.numpy()
        scores_np = scores.numpy()
        dets = np.hstack((boxes_np, scores_np))
        keep = nms(dets, self._cfg['TEST.RPN_NMS_THRESH'])
        keep = tf.convert_to_tensor(keep, dtype=tf.int64)
        if self._cfg['TEST.RPN_POST_NMS_TOP_N'] > 0:
            keep = keep[:self._cfg['TEST.RPN_POST_NMS_TOP_N']]
        
        boxes = tf.gather(boxes, keep,axis=0)
        ellipses = tf.gather(ellipses, keep,axis=0)
        scores = tf.reshape(tf.gather(scores, keep),[-1])
        
        return (boxes, ellipses, scores)
