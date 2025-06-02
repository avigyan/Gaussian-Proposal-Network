import tensorflow as tf
import numpy as np

np.random.seed(0)

from model.generate_anchor import generate_anchors
from model.bbox_transform import bbox_overlaps_batch, bbox_transform

def _unmap(data, count, inds, batch_size, fill=0):  
    """
    Unmap a subset of items (data) back to the original set of items (of size count).
    
    Args:
        data: tf.Tensor, shape either (batch_size, len(inds)) or (batch_size, len(inds), D)
        count: int, total number of items to map back into
        inds: 1D list or tf.Tensor of indices where data should be placed
        batch_size: int
        fill: scalar value to fill unmapped locations (default 0)
        
    Returns:
        ret: tf.Tensor of shape (batch_size, count) or (batch_size, count, D)
    """

    inds = tf.convert_to_tensor(inds, dtype=tf.int64)
    
    if len(data.shape) == 2:
        ret = tf.fill([batch_size, count], tf.cast(fill, data.dtype))
        
        batch_idx = tf.repeat(tf.range(batch_size), repeats=tf.shape(inds)[0])
        col_idx = tf.tile(inds, [batch_size])
        scatter_idx = tf.stack([batch_idx, col_idx], axis=1)
        
        ret = tf.tensor_scatter_nd_update(ret, scatter_idx, tf.reshape(data, [-1]))
    
    else:
        # data shape: (batch_size, len(inds), D)
        D = data.shape[2]
        ret = tf.fill([batch_size, count, D], tf.cast(fill, data.dtype))
        
        batch_idx = tf.repeat(tf.range(batch_size), repeats=tf.shape(inds)[0])
        col_idx = tf.tile(inds, [batch_size])
        
        # For 3D indexing, we need to repeat batch and col indices for each D
        # So we expand dims for feature dimension
        batch_idx_exp = tf.expand_dims(batch_idx, axis=1)  # (batch_size*len(inds), 1)
        col_idx_exp = tf.expand_dims(col_idx, axis=1)      # same shape
        
        # Create indices for all D features:
        feature_idx = tf.range(D, dtype=tf.int64)          # (D,)
        feature_idx_tiled = tf.tile(tf.expand_dims(feature_idx, 0), [tf.shape(batch_idx)[0], 1])  # (batch_size*len(inds), D)
        
        batch_idx_tiled = tf.tile(batch_idx_exp, [1, D])
        col_idx_tiled = tf.tile(col_idx_exp, [1, D])
        
        scatter_idx = tf.stack([batch_idx_tiled, col_idx_tiled, feature_idx_tiled], axis=2)
        scatter_idx = tf.reshape(scatter_idx, [-1, 3])
        
        ret = tf.tensor_scatter_nd_update(ret, scatter_idx, tf.reshape(data, [-1]))
    
    return ret

class BboxTargetLayer(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(BboxTargetLayer, self).__init__()
        self._cfg = dict(cfg)
        self._preprocess()

    def _preprocess(self):  
        # pre-computing stuff for making anchor later
        allowed_border = 0
        im_info = (self._cfg['MAX_SIZE'], self._cfg['MAX_SIZE'])
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
        all_anchors = base_anchors.reshape((1, A, 4)) + \
            shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -allowed_border) &
            (all_anchors[:, 1] >= -allowed_border) &
            (all_anchors[:, 2] < im_info[1] + allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + allowed_border)    # height
        )[0]

        anchors = all_anchors[inds_inside, :]

        self._A = A
        self._feat_height = feat_height
        self._feat_width = feat_width
        self._total_anchors = total_anchors
        self._inds_inside = tf.convert_to_tensor(inds_inside, dtype=tf.int64)
        self._anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

    def call(self, gt_boxes):
        batch_size = tf.shape(gt_boxes)[0]

        labels = tf.fill([batch_size, tf.shape(self._inds_inside)[0]], tf.constant(-1, dtype=gt_boxes.dtype))

        overlaps = bbox_overlaps_batch(self._anchors, gt_boxes)

        max_overlaps = tf.reduce_max(overlaps, axis=2)
        argmax_overlaps = tf.argmax(overlaps, axis=2)
        gt_max_overlaps = tf.reduce_max(overlaps, axis=1)

        # assign bg labels first so that positive labels can clobber them
        labels = tf.where(max_overlaps < self._cfg['TRAIN.RPN_NEGATIVE_OVERLAP'], tf.zeros_like(labels), labels)

        # mark max overlap of 0, which are padded gt_boxes
        gt_max_overlaps = tf.where(gt_max_overlaps == 0, 1e-5, gt_max_overlaps)
        # mark the max overlap of each ground truth

        #labels[max_overlaps >= self._cfg['TRAIN.RPN_POSITIVE_OVERLAP']] = 1
        keep = tf.reduce_sum(tf.cast(overlaps == tf.expand_dims(gt_max_overlaps, axis=1), tf.float32), axis=2)
        if tf.reduce_sum(keep) > 0:
            labels = tf.where(keep > 0, tf.ones_like(labels), labels)

        labels = tf.where(max_overlaps >= self._cfg['TRAIN.RPN_POSITIVE_OVERLAP'], 1, labels)

        # subsample positive labels if we have too many
        num_fg = int(self._cfg['TRAIN.RPN_FG_FRACTION'] *
                     self._cfg['TRAIN.RPN_BATCHSIZE'])

        sum_fg = tf.reduce_sum(tf.cast(labels == 1, tf.int64), axis=1)
        sum_bg = tf.reduce_sum(tf.cast(labels == 0, tf.int64), axis=1)

        bbox_targets = tf.fill([batch_size, tf.shape(self._inds_inside)[0], 4], 0)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = tf.reshape(tf.where(labels[i] == 1),[-1])
                rand_num = tf.random.shuffle(tf.range(tf.shape(fg_inds)[0], dtype=tf.int64))
                disable_inds = tf.gather(fg_inds, rand_num[:tf.shape(fg_inds)[0] - num_fg])
                
                # Prepare indices for scatter update
                batch_idx = tf.fill([tf.shape(disable_inds)[0], 1], i)        # shape (num_disable_inds, 1)
                col_idx = tf.expand_dims(disable_inds, axis=1)                # shape (num_disable_inds, 1)
                indices = tf.concat([batch_idx, col_idx], axis=1)             # shape (num_disable_inds, 2)

                # Values to assign (-1)
                updates = tf.fill([tf.shape(disable_inds)[0]], tf.constant(-1, dtype=labels.dtype))

                labels = tf.tensor_scatter_nd_update(labels, indices, updates)


            num_bg = self._cfg['TRAIN.RPN_BATCHSIZE'] - tf.reduce_sum(tf.cast(labels == 1, tf.int64), axis=1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = tf.reshape(tf.where(labels[i] == 0),[-1])
                rand_num = perm = tf.random.shuffle(tf.range(tf.shape(bg_inds)[0], dtype=tf.int64))
                disable_inds = tf.gather(bg_inds, rand_num[:tf.shape(bg_inds)[0] - num_bg])
                
                # Create indices for scatter update: shape (num_disable_inds, 2)
                batch_indices = tf.fill([tf.shape(disable_inds)[0], 1], i)
                col_indices = tf.expand_dims(disable_inds, axis=1)
                indices = tf.concat([batch_indices, col_indices], axis=1)

                # Values to assign
                updates = tf.fill([tf.shape(disable_inds)[0]], tf.constant(-1, dtype=labels.dtype))

                # Update labels tensor
                labels = tf.tensor_scatter_nd_update(labels, indices, updates)


                # Compute bbox transform for batch i
                selected_gt_boxes = tf.gather(gt_boxes[i], argmax_overlaps[i])[:, :4]  # shape (num_anchors, 4)
                transformed = bbox_transform(self._anchors, selected_gt_boxes)        # shape (num_anchors, 4)

                # Prepare indices to update bbox_targets[i]
                indices = tf.expand_dims(tf.range(tf.shape(transformed)[0]), axis=1)  # shape (num_anchors, 1)

                # Update bbox_targets[i]
                bbox_targets_i = tf.tensor_scatter_nd_update(bbox_targets[i], indices, transformed)

                # Replace bbox_targets[i] in bbox_targets
                batch_idx = tf.fill([tf.shape(transformed)[0], 1], i)
                full_indices = tf.concat([batch_idx, indices], axis=1)  # shape (num_anchors, 2)
                bbox_targets = tf.tensor_scatter_nd_update(bbox_targets, full_indices, transformed)


        # map up to original set of anchors
        labels = _unmap(
            labels, self._total_anchors, self._inds_inside, batch_size,
            fill=-1)
        bbox_targets = _unmap(
            bbox_targets, self._total_anchors, self._inds_inside, batch_size,
            fill=0)

        
        labels = tf.reshape(labels, [batch_size, self._feat_height, self._feat_width, self._A, 1])
        bbox_targets = tf.reshape(bbox_targets, [batch_size, self._feat_height, self._feat_width, self._A, 4])

        return (labels, bbox_targets)
