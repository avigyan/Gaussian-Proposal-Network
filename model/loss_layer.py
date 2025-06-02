import tensorflow as tf
import numpy as np

from model.generate_anchor import generate_anchors


class LossCls(tf.keras.layers.Layer):

    def __init__(self):
        super(LossCls, self).__init__()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

    def call(self, out_cls, labels):
        label_idcs = tf.where(tf.not_equal(tf.reshape(labels, [-1]), -1))
        label_idcs = tf.reshape(label_idcs, [-1])
        out_cls_keep = tf.gather(tf.reshape(out_cls, [-1, 2]), label_idcs) 
        labels_keep = tf.gather(tf.reshape(labels, [-1]), label_idcs) 

        loss = self.cls_loss(labels_keep, out_cls_keep)

        return loss


class LossBbox(tf.keras.layers.Layer): 

    def __init__(self):
        super(LossBbox, self).__init__()
        self.bbox_loss = tf.keras.losses.Huber()

    def call(self, out_bbox, labels, bbox_targets):
        pos_idcs = tf.where(tf.equal(tf.reshape(labels, [-1]), 1))
        pos_idcs = tf.reshape(pos_idcs, [-1])
        
        out_bbox_keep = tf.gather(tf.reshape(out_bbox, [-1, 4]), pos_idcs)
        bbox_targets_keep = tf.gather(tf.reshape(bbox_targets, [-1, 4]), pos_idcs)

        loss = self.bbox_loss(bbox_targets_keep, out_bbox_keep) 
        return loss


class LossEllipseSL1(tf.keras.layers.Layer): 

    def __init__(self):
        super(LossEllipseSL1, self).__init__()
        self.ellipse_loss = tf.keras.losses.Huber()

    
    def call(self, out_ellipse, labels, ellipse_targets):
        pos_idcs = tf.where(tf.equal(tf.reshape(labels, [-1]), 1))
        pos_idcs = tf.reshape(pos_idcs, [-1])
        out_ellipse_keep = tf.gather(tf.reshape(out_ellipse, [-1, 5]), pos_idcs)
        ellipse_targets_keep = tf.gather(tf.reshape(ellipse_targets, [-1, 5]), pos_idcs)

        loss = self.ellipse_loss(ellipse_targets_keep, out_ellipse_keep)

        return loss


class LossEllipseKLD(tf.keras.layers.Layer):

    def __init__(self, cfg):
        super(LossEllipseKLD, self).__init__()
        self._cfg = dict(cfg)
        self._preprocess()

    def _preprocess(self):
        # pre-computing stuff for making anchor later
        base_anchors = generate_anchors(
            base_size=self._cfg['RPN_FEAT_STRIDE'],
            ratios=[1],
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

        
        self.anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    
    def call(self, out_ellipse, labels, ellipse_targets):
        batch_size = tf.shape(out_ellipse)[0]
        anchors = tf.tile(self.anchors, [batch_size, 1])

        pos_idcs = pos_idcs = tf.where(tf.equal(tf.reshape(labels, [-1]), 1))
        pos_idcs = tf.reshape(pos_idcs, [-1])
        out_ellipse_keep = tf.gather(tf.reshape(out_ellipse, [-1, 5]), pos_idcs)
        ellipse_targets_keep = tf.gather(tf.reshape(ellipse_targets, [-1, 5]), pos_idcs)
        anchors_keep = tf.gather(tf.reshape(anchors, [-1, 4]), pos_idcs)

        sigmas = (anchors_keep[:, 2] - anchors_keep[:, 0] + 1.0) / 2

        dx_o = out_ellipse_keep[:, 0]
        dy_o = out_ellipse_keep[:, 1]
        dl_o = out_ellipse_keep[:, 2]
        ds_o = out_ellipse_keep[:, 3]
        theta_o = tf.atan(out_ellipse_keep[:, 4])
        l_o = tf.exp(dl_o) * sigmas
        s_o = tf.exp(ds_o) * sigmas

        dx_t = ellipse_targets_keep[:, 0]
        dy_t = ellipse_targets_keep[:, 1]
        dl_t = ellipse_targets_keep[:, 2]
        ds_t = ellipse_targets_keep[:, 3]
        theta_t = tf.atan(ellipse_targets_keep[:, 4])

        dx = 2 * sigmas * (dx_o - dx_t)
        dy = 2 * sigmas * (dy_o - dy_t)
        dtheta = theta_o - theta_t

        trace = (tf.cos(dtheta) * tf.exp(dl_t - dl_o))**2 + \
                (tf.cos(dtheta) * tf.exp(ds_t - ds_o))**2 + \
                (tf.sin(dtheta) * tf.exp(dl_t - ds_o))**2 + \
                (tf.sin(dtheta) * tf.exp(ds_t - dl_o))**2 

        dist = ((tf.cos(theta_o) * dx + tf.sin(theta_o) * dy) / l_o)**2 + \
               ((tf.cos(theta_o) * dy - tf.sin(theta_o) * dx) / s_o)**2 

        determinant = 2 * (dl_o - dl_t) + 2 * (ds_o - ds_t)

        kld = (trace + dist + determinant - 2) / 2

        return tf.reduce_mean(kld)
