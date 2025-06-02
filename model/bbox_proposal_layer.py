import tensorflow as tf
import numpy as np

np.random.seed(0)

from model.generate_anchor import generate_anchors
from model.bbox_transform import bbox_transform_inv, clip_boxes
#from nms.cpu_nms import cpu_nms
#from nms.gpu_nms import gpu_nms
from nms import nms

def _filter_boxes(boxes, min_size): # ok
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = tf.reshape(tf.where((ws >= min_size) & (hs >= min_size)),[-1])

    return keep


class BboxProposalLayer(nn.Module):
    def __init__(self, cfg):
        super(BboxProposalLayer, self).__init__()
        self._cfg = dict(cfg)
        self._preprocess()

    def _preprocess(self): 
        # pre-computing stuff for making anchor later
        self._im_info = (self._cfg['MAX_SIZE'], self._cfg['MAX_SIZE'])
        base_anchors = generate_anchors(
            base_size=self._cfg['RPN_FEAT_STRIDE'],
            ratios=self._cfg['ANCHOR_RATIOS'],
            scales=np.array(self._cfg['ANCHOR_SCALES'], dtype=np.float32))
        num_anchors = base_anchors.shape[0]
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
        anchors = base_anchors.reshape((1, A, 4)) + \
            shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        self._feat_height = feat_height
        self._feat_width = feat_width
        self._anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)


    def call(self, out_cls, out_bbox):
        """
        out_cls: (feat_height, feat_width, anchors, 2) FloatVariable
        out_bbox: (feat_height, feat_width, anchors, 4) FloatVariable
        """
        scores = tf.nn.softmax(out_cls, axis=-1)[..., 1]
        scores = tf.stop_gradient(scores)
        scores = tf.reshape(scores, [-1, 1])
        out_bbox = tf.stop_gradient(out_bbox)
        bbox_deltas = tf.reshape(out_bbox, [-1, 4])

        # 1. Generate proposals from bbox deltas and shifted anchors
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(self._anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, self._im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTICE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, self._cfg['TEST.RPN_MIN_SIZE'])
        proposals = tf.gather(proposals, keep,axis=0)
        scores = tf.gather(scores, keep)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = tf.argsort(tf.reshape(scores, [-1]), direction='DESCENDING')
        if self._cfg['TEST.RPN_PRE_NMS_TOP_N'] > 0:
            order = order[:self._cfg['TEST.RPN_PRE_NMS_TOP_N']]
        proposals = tf.gather(proposals, order,axis=0)
        scores = tf.gather(scores, order)

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        #if self._cfg['USE_GPU_NMS']:
        #    nms = gpu_nms
        #else:
        #    nms = cpu_nms
        scores = tf.reshape(scores, [-1, 1]) # Optional Safety Check
        dets = tf.concat([proposals, scores], axis=1).numpy()
        keep = nms(dets, self._cfg['TEST.RPN_NMS_THRESH'])
        keep = tf.convert_to_tensor(keep, dtype=tf.int64)
        if self._cfg['TEST.RPN_POST_NMS_TOP_N'] > 0:
            keep = keep[:self._cfg['TEST.RPN_POST_NMS_TOP_N']]
    
        proposals = tf.gather(proposals, keep,axis=0)
        scores = tf.reshape(tf.gather(scores, keep),[-1])

        return proposals, scores
