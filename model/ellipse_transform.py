import tensorflow as tf
import numpy as np


def ellipse2box(ellipses, pad): 
    """
    ellipses: (N, 8) ndarray of float
    pad: int, padded margin with respect to the corner of ellipse
    """
    xs = tf.stack(
        (ellipses[:, 0], ellipses[:, 2], ellipses[:, 4], ellipses[:, 6]), 1)
    ys = tf.stack(
        (ellipses[:, 1], ellipses[:, 3], ellipses[:, 5], ellipses[:, 7]), 1)
    x_min = tf.reduce_min(xs, axis=1)
    x_max = tf.reduce_max(xs, axis=1)
    y_min = tf.reduce_min(ys, axis=1)
    y_max = tf.reduce_max(ys, axis=1)

    boxes = tf.zeros((tf.shape(ellipses)[0], 4), dtype=ellipses.dtype)

    
    # x1
    boxes = tf.tensor_scatter_nd_update(boxes, [[i, 0] for i in range(boxes.shape[0])], x_min - pad)

    # y1
    boxes = tf.tensor_scatter_nd_update(boxes, [[i, 1] for i in range(boxes.shape[0])], y_min - pad)

    # x2
    boxes = tf.tensor_scatter_nd_update(boxes, [[i, 2] for i in range(boxes.shape[0])], x_max + pad)

    # y2
    boxes = tf.tensor_scatter_nd_update(boxes, [[i, 3] for i in range(boxes.shape[0])], y_max + pad)

    return boxes


def ellipse_mask(ellipses, im_info):
    """
    ellipses: (N, 8) ndarray of float, ellipses
    im_info: 2d int tuple, (width, height)
    mask: (width, height, N) ndarray of binary mask representing if pixels
        are within the corresponding ellipse.
    """
    N = tf.shape(ellipses)[0]
    width, height = im_info

    longs_x = (tf.abs(ellipses[:, 2] - ellipses[:, 0]) + 1.0) / 2.0
    longs_y = (tf.abs(ellipses[:, 3] - ellipses[:, 1]) + 1.0) / 2.0
    longs = tf.sqrt(longs_x**2 + longs_y**2)
    shorts_x = (tf.abs(ellipses[:, 6] - ellipses[:, 4]) + 1.0) / 2.0
    shorts_y = (tf.abs(ellipses[:, 7] - ellipses[:, 5]) + 1.0) / 2.0
    shorts = tf.sqrt(shorts_x**2 + shorts_y**2)
    ctr_x = tf.minimum(ellipses[:, 0], ellipses[:, 2]) + (longs_x - 0.5)
    ctr_y = tf.minimum(ellipses[:, 1], ellipses[:, 3]) + (longs_y - 0.5)
    # tan(theta), theta is the angle between x > 0 axis and long > 0 axis.
    sign = tf.sign(
        (ellipses[:, 2] - ellipses[:, 0]) * (ellipses[:, 3] - ellipses[:, 1]))
    tan = sign * longs_y / longs_x
    theta = tf.atan(tan)

    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    x = tf.convert_to_tensor(x.ravel(), dtype=ellipses.dtype)
    y = tf.convert_to_tensor(y.ravel(), dtype=ellipses.dtype)

    x, y = tf.reshape(x, [-1, 1]), tf.reshape(y, [-1, 1])

    ctr_x, ctr_y = tf.reshape(ctr_x, [1, -1]), tf.reshape(ctr_y, [1, -1])
    longs, shorts = tf.reshape(longs, [1, -1]), tf.reshape(shorts, [1, -1])
    theta = tf.reshape(theta, [1, -1])#theta.view(1, -1)

    dx = x - ctr_x
    dy = y - ctr_y

    dist = ((tf.cos(theta) * dx + tf.sin(theta) * dy) / longs)**2 + \
           ((tf.cos(theta) * dy - tf.sin(theta) * dx) / shorts)**2

    mask = tf.reshape(dist <= 1.0, [width, height, N])

    return mask


def ellipse_overlaps(qr_ellipses, gt_ellipses, im_info):
    """
    qr_ellipses: (N, 8) ndarray of float, query ellipses
    gt_ellipses: (K, 8) ndarray of float, ground truth ellipses
    im_info: 2d int tuple, (width, height)
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = tf.shape(qr_ellipses)[0]
    K = tf.shape(gt_ellipses)[0]
    width, height = im_info

    qr_mask = ellipse_mask(qr_ellipses, im_info)
    gt_mask = ellipse_mask(gt_ellipses, im_info)

    qr_mask = tf.cast(tf.transpose(tf.reshape(qr_mask, [-1, N]), [1, 0]), tf.float32)
    gt_mask = tf.cast(tf.reshape(gt_mask, [-1, K]), tf.float32)

    I = tf.matmul(qr_mask, gt_mask)
    U = width * height - tf.matmul(1 - qr_mask, 1 - gt_mask)
    overlaps = I / U

    return overlaps


def ellipse_transform(ex_rois, gt_rois):
    """
    ex_rois: (N, 4) ndarray of float, anchors
    gt_rois: (N, 8) ndarray of float, ellipses
    """
    # assuming anchors have equal width and height, which is defined as sigma*2
    ex_sigmas = (ex_rois[:, 2] - ex_rois[:, 0] + 1.0) / 2
    ex_ctr_x = ex_rois[:, 0] + (ex_sigmas - 0.5)
    ex_ctr_y = ex_rois[:, 1] + (ex_sigmas - 0.5)

    gt_longs_x = (tf.abs(gt_rois[:, 2] - gt_rois[:, 0]) + 1.0) / 2.0
    gt_longs_y = (tf.abs(gt_rois[:, 3] - gt_rois[:, 1]) + 1.0) / 2.0
    gt_longs = tf.sqrt(gt_longs_x**2 + gt_longs_y**2)
    gt_shorts_x = (tf.abs(gt_rois[:, 6] - gt_rois[:, 4]) + 1.0) / 2.0
    gt_shorts_y = (tf.abs(gt_rois[:, 7] - gt_rois[:, 5]) + 1.0) / 2.0
    gt_shorts = tf.sqrt(gt_shorts_x**2 + gt_shorts_y**2)
    gt_ctr_x = tf.minimum(gt_rois[:, 0], gt_rois[:, 2]) + (gt_longs_x - 0.5)
    gt_ctr_y = tf.minimum(gt_rois[:, 1], gt_rois[:, 3]) + (gt_longs_y - 0.5)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_sigmas * 2)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_sigmas * 2)
    targets_dl = tf.math.log(gt_longs / ex_sigmas)
    targets_ds = tf.math.log(gt_shorts / ex_sigmas)
    # tan(theta), theta is the angle between x > 0 axis and long > 0 axis.
    targets_sign = tf.sign(
        (gt_rois[:, 2] - gt_rois[:, 0]) * (gt_rois[:, 3] - gt_rois[:, 1]))
    targets_tan = targets_sign * gt_longs_y / gt_longs_x

    targets = tf.stack(
        (targets_dx, targets_dy, targets_dl, targets_ds, targets_tan), axis = 1)

    return targets


def ellipse_transform_inv(ellipses, deltas):
    sigmas = (ellipses[:, 2] - ellipses[:, 0] + 1.0) / 2
    ctr_x = ellipses[:, 0] + (sigmas - 0.5)
    ctr_y = ellipses[:, 1] + (sigmas - 0.5)

    dx = deltas[:, 0::5]
    dy = deltas[:, 1::5]
    dl = deltas[:, 2::5]
    ds = deltas[:, 3::5]
    tan = deltas[:, 4::5]
    theta = tf.atan(tan)
    sign = tf.sign(theta)

    #print(dx, sigmas, ctr_x)
    pred_ctr_x = dx * tf.expand_dims(sigmas, axis=1) * 2 + tf.expand_dims(ctr_x, axis=1)
    pred_ctr_y = dy * tf.expand_dims(sigmas, axis=1) * 2 + tf.expand_dims(ctr_y, axis=1)

    pred_l = tf.exp(dl) * tf.expand_dims(sigmas, axis=1)
    pred_s = tf.exp(ds) * tf.expand_dims(sigmas, axis=1)
    pred_l_x = tf.abs(pred_l * tf.cos(theta))
    pred_l_y = tf.abs(pred_l * tf.sin(theta))
    pred_s_x = tf.abs(pred_s * tf.sin(theta))
    pred_s_y = tf.abs(pred_s * tf.cos(theta))

    pred_elipses = tf.zeros((tf.shape(deltas)[0], 8), dtype=deltas.dtype)

    
    #print(pred_elipses.shape)
    #print(pred_ctr_x.shape)
    #print(pred_l_x.shape)
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 0] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_x[:, 0] - (pred_l_x[:, 0] - 0.5))
        
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 1] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_y[:, 0] - sign[:, 0] * (pred_l_y[:, 0] - 0.5))
    
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 2] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_x[:, 0] + (pred_l_x[:, 0] - 0.5))
    
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 3] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_y[:, 0] + sign[:, 0] * (pred_l_y[:, 0] - 0.5))
     
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 4] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_x[:, 0] - (pred_s_x[:, 0] - 0.5))
        
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 5] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_y[:, 0] + sign[:, 0] * (pred_s_y[:, 0] - 0.5))
        
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 6] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_x[:, 0] + (pred_s_x[:, 0] - 0.5))
    
    pred_elipses = tf.tensor_scatter_nd_update(pred_elipses, [[i, 7] for i in range(tf.shape(pred_elipses)[0])], pred_ctr_y[:, 0] - sign[:, 0] * (pred_s_y[:, 0] - 0.5))
    
    return pred_elipses
