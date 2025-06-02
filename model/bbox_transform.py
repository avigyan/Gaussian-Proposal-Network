import tensorflow as tf

def bbox_overlaps(anchors, gt_boxes): # ok
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    
    N = tf.shape(anchors)[0]
    K = tf.shape(gt_boxes)[0]

    gt_boxes_area = tf.reshape(((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
                     (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)),(1,K))

    anchors_area = tf.reshape(((anchors[:, 2] - anchors[:, 0] + 1) *
                    (anchors[:, 3] - anchors[:, 1] + 1)),(N,1))

    boxes = tf.expand_dims(anchors, axis=1)
    boxes = tf.tile(boxes, [1, K, 1])

    query_boxes = tf.expand_dims(gt_boxes, axis=0)
    query_boxes = tf.tile(query_boxes, [N, 1, 1])
    
    iw = tf.minimum(boxes[:, :, 2], query_boxes[:, :, 2]) - \
         tf.maximum(boxes[:, :, 0], query_boxes[:, :, 0]) + 1
    iw = tf.maximum(iw, 0)

    ih = tf.minimum(boxes[:, :, 3], query_boxes[:, :, 3]) - \
         tf.maximum(boxes[:, :, 1], query_boxes[:, :, 1]) + 1
    ih = tf.maximum(ih, 0)

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def bbox_overlaps_batch(anchors, gt_boxes): # ok
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (B, K, 4) ndarray of float, where B is batch_size
    overlaps: (B, N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = tf.shape(gt_boxes)[0]
    N = tf.shape(anchors)[0]
    K = tf.shape(gt_boxes)[1]

    anchors = tf.tile(tf.expand_dims(anchors, axis=0), [batch_size, 1, 1])
    gt_boxes = gt_boxes[:, :, :4]

    
    gt_boxes_area = ((gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1) *
                     (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1))[:, tf.newaxis, :]

    anchors_area = ((anchors[:, :, 2] - anchors[:, :, 0] + 1) *
                    (anchors[:, :, 3] - anchors[:, :, 1] + 1))[:, :, tf.newaxis]

    
    boxes = tf.expand_dims(anchors, axis=2)
    boxes = tf.tile(boxes, [1, 1, K, 1])
    query_boxes = tf.expand_dims(gt_boxes, axis=1)
    query_boxes = tf.tile(query_boxes, [1, N, 1, 1])


    
    iw = tf.minimum(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) - \
         tf.maximum(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1
    iw = tf.maximum(iw, 0)

    ih = tf.minimum(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) - \
         tf.maximum(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1
    ih = tf.maximum(ih, 0)

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def bbox_transform(ex_rois, gt_rois): # ok
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    #print(gt_ctr_x.shape, ex_ctr_x.shape, ex_widths.shape)
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    
    targets_dw = tf.math.log(gt_widths / ex_widths)
    targets_dh = tf.math.log(gt_heights / ex_heights)

    targets = tf.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)

    return targets


def bbox_transform_inv(boxes, deltas): # ok
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * tf.expand_dims(widths, axis=1) + tf.expand_dims(ctr_x, axis=1)
    pred_ctr_y = dy * tf.expand_dims(heights, axis=1) + tf.expand_dims(ctr_y, axis=1)
    pred_w = tf.exp(dw) * tf.expand_dims(widths, axis=1)
    pred_h = tf.exp(dh) * tf.expand_dims(heights, axis=1)

    
    pred_boxes = tf.identity(deltas)
    # x1
    pred_boxes = tf.tensor_scatter_nd_update(pred_boxes, [(i, j) for i in range(pred_boxes.shape[0]) for j in range(0, pred_boxes.shape[1], 4)], tf.reshape(pred_ctr_x - 0.5 * (pred_w - 1.0), [-1]))
    # y1
    pred_boxes = tf.tensor_scatter_nd_update(pred_boxes, [(i, j) for i in range(pred_boxes.shape[0]) for j in range(1, pred_boxes.shape[1], 4)], tf.reshape(pred_ctr_y - 0.5 * (pred_h - 1.0), [-1]))
    # x2
    pred_boxes = tf.tensor_scatter_nd_update(pred_boxes, [(i, j) for i in range(pred_boxes.shape[0]) for j in range(2, pred_boxes.shape[1], 4)], tf.reshape(pred_ctr_x + 0.5 * (pred_w - 1.0), [-1]))
    # y2
    pred_boxes = tf.tensor_scatter_nd_update(pred_boxes, [(i, j) for i in range(pred_boxes.shape[0]) for j in range(3, pred_boxes.shape[1], 4)], tf.reshape(pred_ctr_y + 0.5 * (pred_h - 1.0), [-1]))
    
    return pred_boxes


def clip_boxes(boxes, im_shape): # ok
    """
    Clip boxes to image boundaries.
    """

    
    boxes = tf.identity(boxes)
    # x1 >= 0
    boxes = tf.tensor_scatter_nd_update(boxes, [(i, j) for i in range(boxes.shape[0]) for j in range(0, boxes.shape[1], 4)], tf.reshape(tf.clip_by_value(boxes[:, 0::4], 0, im_shape[1] - 1), [-1]))

    # y1 >= 0
    boxes = tf.tensor_scatter_nd_update(boxes, [(i, j) for i in range(boxes.shape[0]) for j in range(1, boxes.shape[1], 4)], tf.reshape(tf.clip_by_value(boxes[:, 1::4], 0, im_shape[0] - 1), [-1]))

    # x2 < im_shape[1]
    boxes = tf.tensor_scatter_nd_update(boxes, [(i, j) for i in range(boxes.shape[0]) for j in range(2, boxes.shape[1], 4)], tf.reshape(tf.clip_by_value(boxes[:, 2::4], 0, im_shape[1] - 1), [-1]))

    # y2 < im_shape[0]
    boxes = tf.tensor_scatter_nd_update(boxes, [(i, j) for i in range(boxes.shape[0]) for j in range(3, boxes.shape[1], 4)], tf.reshape(tf.clip_by_value(boxes[:, 3::4], 0, im_shape[0] - 1), [-1]))

    return boxes
