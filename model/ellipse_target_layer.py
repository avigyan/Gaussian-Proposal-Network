import tensorflow as tf
import numpy as np

from model.generate_anchor import generate_anchors
from model.bbox_transform import bbox_transform, bbox_overlaps_batch
from model.ellipse_transform import ellipse_transform

np.random.seed(0)

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of items (data) back to the original set of items (of size count) """
    inds = tf.convert_to_tensor(inds, dtype=tf.int64)
    dtype = data.dtype
    
    # Determine output shape based on data dimensions
    data_rank = tf.rank(data)
    output_shape = tf.cond(
        tf.equal(data_rank, 2),
        lambda: tf.stack([batch_size, count]),
        lambda: tf.stack([batch_size, count, tf.shape(data)[2]])
    )
    
    # Create base tensor filled with fill value
    base = tf.cast(tf.fill(output_shape, fill), dtype=dtype)
    
    # Generate indices and updates based on data dimensions
    def handle_2d():
        batch_indices = tf.repeat(tf.range(batch_size, dtype=tf.int64), tf.shape(inds)[0])
        scatter_inds = tf.stack([batch_indices, tf.tile(inds, [batch_size])], axis=1)
        return tf.tensor_scatter_nd_update(base, scatter_inds, tf.reshape(data, [-1]))
    
    def handle_3d():
        features = tf.shape(data)[2]
        batch_indices = tf.repeat(
            tf.range(batch_size, dtype=tf.int64), 
            tf.shape(inds)[0] * features
        )
        expanded_inds = tf.repeat(inds, features)
        scatter_inds = tf.stack([
            batch_indices,
            tf.tile(expanded_inds, [batch_size]),
            tf.tile(tf.range(features, dtype=tf.int64), [batch_size * tf.shape(inds)[0]])
        ], axis=1)
        return tf.tensor_scatter_nd_update(base, scatter_inds, tf.reshape(data, [-1]))
    
    return tf.cond(
        tf.equal(data_rank, 2),
        handle_2d,
        handle_3d
    )

class EllipseTargetLayer(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(EllipseTargetLayer, self).__init__()
        self.cfg = cfg
        self._preprocess()

    def _preprocess(self):
        allowed_border = 0
        im_info = (self.cfg['MAX_SIZE'], self.cfg['MAX_SIZE'])
        base_anchors = generate_anchors(
            base_size=self.cfg['RPN_FEAT_STRIDE'],
            ratios=[1],
            scales=np.array(self.cfg['ANCHOR_SCALES'], dtype=np.float32))
        num_anchors = base_anchors.shape[0]
        feat_stride = self.cfg['RPN_FEAT_STRIDE']
        feat_width = self.cfg['MAX_SIZE'] // feat_stride
        feat_height = feat_width

        shift_x = np.arange(0, feat_width) * feat_stride
        shift_y = np.arange(0, feat_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), 
                            shift_x.ravel(), shift_y.ravel())).T
        
        A = num_anchors
        K = shifts.shape[0]
        all_anchors = (base_anchors.reshape((1, A, 4)) + \
                     shifts.reshape((1, K, 4)).transpose(1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = K * A

        inds_inside = np.where(
            (all_anchors[:, 0] >= -allowed_border) &
            (all_anchors[:, 1] >= -allowed_border) &
            (all_anchors[:, 2] < im_info[1] + allowed_border) &
            (all_anchors[:, 3] < im_info[0] + allowed_border)
        )[0]

        anchors = all_anchors[inds_inside, :]

        self._A = A
        self._feat_height = feat_height
        self._feat_width = feat_width
        self._total_anchors = total_anchors
        self._inds_inside = tf.constant(inds_inside, dtype=tf.int64)
        self._anchors = tf.constant(anchors, dtype=tf.float32)

    def call(self, gt_boxes, gt_ellipses):
        batch_size = tf.shape(gt_boxes)[0]
        labels = tf.fill(
            [batch_size, tf.shape(self._inds_inside)[0]], tf.constant(-1, dtype=gt_boxes.dtype))

        overlaps = bbox_overlaps_batch(self._anchors, gt_boxes)
        max_overlaps = tf.reduce_max(overlaps, axis=2)
        argmax_overlaps = tf.argmax(overlaps, axis=2, output_type=tf.int64)
        gt_max_overlaps = tf.reduce_max(overlaps, axis=1)

        labels = tf.where(
            max_overlaps < self.cfg['TRAIN.RPN_NEGATIVE_OVERLAP'], 
            tf.zeros_like(labels), labels)
        gt_max_overlaps = tf.where(
            tf.equal(gt_max_overlaps, 0), 1e-5, gt_max_overlaps)

        expanded_gt_max = tf.expand_dims(gt_max_overlaps, 1)
        
        keep = tf.reduce_sum(tf.cast(tf.equal(overlaps, expanded_gt_max), tf.int64), axis=2)
        
        if tf.reduce_sum(keep) > 0:
            labels = tf.where(keep > 0, tf.ones_like(labels), labels)

        labels = tf.where(
            max_overlaps >= self.cfg['TRAIN.RPN_POSITIVE_OVERLAP'], 
            tf.ones_like(labels), labels)

        num_fg = int(self.cfg['TRAIN.RPN_FG_FRACTION'] * 
                  self.cfg['TRAIN.RPN_BATCHSIZE'])
        sum_fg = tf.reduce_sum(tf.cast(labels == 1, tf.int64), axis=1)
        sum_bg = tf.reduce_sum(tf.cast(labels == 0, tf.int64), axis=1)

        bbox_targets = tf.zeros(
            [batch_size, tf.shape(self._inds_inside)[0], 4], dtype=gt_boxes.dtype) 
        ellipse_targets = tf.zeros(
            [batch_size, tf.shape(self._inds_inside)[0], 5], dtype=gt_ellipses.dtype) 

        for i in tf.range(batch_size):
            if sum_fg[i] > num_fg:
                fg_mask = labels[i] == 1
                fg_inds = tf.reshape(tf.where(fg_mask),[-1])#[:, 0]
                rand_idx = tf.random.shuffle(tf.range(tf.shape(fg_inds)[0], dtype=tf.int64))
                disable_inds = tf.gather(tf.cast(fg_inds,dtype = tf.int32), tf.cast(rand_idx[:tf.shape(fg_inds)[0] - tf.cast(num_fg,dtype = tf.int32)],dtype = tf.int32))
                
                batch_index = i
                batch_indices = tf.fill(tf.shape(disable_inds), batch_index)
                scatter_indices = tf.stack([batch_indices, disable_inds], axis=1)

                # Create updates filled with -1
                updates = tf.fill(tf.shape(disable_inds), tf.constant(-1, dtype=labels.dtype))

                # Update labels tensor
                labels = tf.tensor_scatter_nd_update(labels, scatter_indices, updates)

            num_bg = self.cfg['TRAIN.RPN_BATCHSIZE'] - tf.reduce_sum(tf.cast(labels == 1, tf.int64), axis=1)[i]

            if sum_bg[i] > num_bg:
                bg_mask = labels[i] == 0
                bg_inds = tf.reshape(tf.where(bg_mask),[-1])#[:, 0]
                rand_idx = tf.random.shuffle(tf.range(tf.shape(bg_inds)[0], dtype=tf.int64)) 
                #print(bg_inds.dtype, rand_idx.dtype, num_bg.dtype, tf.shape(bg_inds)[0].dtype)
                disable_inds = tf.gather(tf.cast(bg_inds,dtype = tf.int32), tf.cast(rand_idx[:tf.shape(bg_inds)[0] - tf.cast(num_bg,dtype = tf.int32)],dtype = tf.int32))
                
                batch_index = i
                batch_indices = tf.fill(tf.shape(disable_inds), batch_index)
                scatter_indices = tf.stack([batch_indices, disable_inds], axis=1)

                updates = tf.fill(tf.shape(disable_inds), tf.constant(-1, dtype=labels.dtype))

                labels = tf.tensor_scatter_nd_update(labels, scatter_indices, updates)


            # Compute the bbox transform result for index i
            update = bbox_transform(self._anchors, tf.gather(gt_boxes[i, :, :4], argmax_overlaps[i])) #, axis = 1

            # Create indices for updating bbox_targets at batch index i, for all anchors
            batch_index = tf.fill([tf.shape(update)[0]], i)  # shape: [num_anchors]
            anchor_indices = tf.range(tf.shape(update)[0], dtype=tf.int64)
            #print(batch_index.dtype, anchor_indices.dtype)
            scatter_indices = tf.stack([batch_index, tf.cast(anchor_indices, dtype = tf.int32)], axis=1)  # shape: [num_anchors, 2]

            # Update bbox_targets
            bbox_targets = tf.tensor_scatter_nd_update(bbox_targets, scatter_indices, update)

            # Compute transformed ellipses for batch i
            update = ellipse_transform(self._anchors, tf.gather(gt_ellipses[i, :, :8], argmax_overlaps[i])) #, axis = 1

            # Prepare scatter indices: batch i, all anchors
            batch_index = tf.fill([tf.shape(update)[0]], i)  # shape: [num_anchors]
            anchor_indices = tf.range(tf.shape(update)[0], dtype=tf.int64)
            scatter_indices = tf.stack([batch_index, tf.cast(anchor_indices, dtype = tf.int32)], axis=1)  # shape: [num_anchors, 2]

            # Update ellipse_targets tensor
            ellipse_targets = tf.tensor_scatter_nd_update(ellipse_targets, scatter_indices, update)




        labels = _unmap(labels, self._total_anchors, 
                       self._inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, self._total_anchors, 
                             self._inds_inside, batch_size, fill=0)
        ellipse_targets = _unmap(ellipse_targets, self._total_anchors, 
                                self._inds_inside, batch_size, fill=0)

        labels = tf.reshape(labels, 
            [batch_size, self._feat_height, self._feat_width, self._A, 1])
        bbox_targets = tf.reshape(bbox_targets, 
            [batch_size, self._feat_height, self._feat_width, self._A, 4])
        ellipse_targets = tf.reshape(ellipse_targets, 
            [batch_size, self._feat_height, self._feat_width, self._A, 5])

        return labels, bbox_targets, ellipse_targets
    