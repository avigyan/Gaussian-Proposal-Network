import tensorflow as tf
import numpy as np


def nms(boxes, scores, overlap=0.7):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        scores: (N) FloatTensor
        boxes: (N, 4) FloatTensor
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
    Return:
        The indices of the kept boxes with respect to N.
    """

    if tf.size(boxes) == 0:
        return tf.zeros((0,), dtype=tf.int64)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    sorted_indices = tf.argsort(scores, direction='DESCENDING')
    
    keep = [] 
    while tf.size(sorted_indices) > 0: 
        i = sorted_indices[0]
        
        keep.append(i.numpy())
        
        if tf.size(sorted_indices) == 1:
            break
        sorted_indices = sorted_indices[1:]
        
        xx1 = tf.maximum(x1[i], tf.gather(x1, sorted_indices))
        yy1 = tf.maximum(y1[i], tf.gather(y1, sorted_indices))
        xx2 = tf.minimum(x2[i], tf.gather(x2, sorted_indices))
        yy2 = tf.minimum(y2[i], tf.gather(y2, sorted_indices))

        w = tf.maximum(0.0, xx2 - xx1)
        h = tf.maximum(0.0, yy2 - yy1)
        inter = w * h

        rem_areas = tf.gather(area, sorted_indices)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        sorted_indices = tf.boolean_mask(sorted_indices, IoU <= overlap)


    return tf.convert_to_tensor(keep, dtype=tf.int64)#keep


def n_proposals(out_cls): 
    
    out_cls_reshaped = tf.reshape(out_cls, [-1, 2])
    vals = tf.reduce_max(out_cls_reshaped, axis=1)
    idcs = tf.argmax(out_cls_reshaped, axis=1)
    n_proposals = tf.reduce_sum(tf.cast(tf.equal(idcs, 1), tf.float32)) / tf.cast(tf.shape(out_cls)[0], tf.float32)

    return n_proposals


def acc(out_cls, labels):
    
    labels_flat = tf.reshape(labels, [-1])
    pos_idcs = tf.reshape(tf.where(labels_flat == 1),[-1])
    neg_idcs = tf.reshape(tf.where(labels_flat == 0),[-1])

    out_cls_flat = tf.reshape(out_cls, [-1, 2])
    out_cls_pos = tf.gather(out_cls_flat, pos_idcs, axis=0)
    prob_pos = tf.nn.softmax(out_cls_pos, axis=1)[:, 1]
    acc_pos = tf.reduce_sum(tf.cast(prob_pos >= 0.5, tf.float32)) / tf.cast(tf.shape(prob_pos)[0], tf.float32)

    out_cls_neg = tf.gather(out_cls_flat, neg_idcs, axis=0)
    prob_neg = tf.nn.softmax(out_cls_neg, axis=1)[:, 0]
    acc_neg = tf.reduce_sum(tf.cast(prob_neg >= 0.5, tf.float32)) / tf.cast(tf.shape(prob_neg)[0], tf.float32)

    return acc_pos, acc_neg


def angle_err(out_ellipse, labels, ellipse_targets):
    labels_flat = tf.reshape(labels, [-1])
    pos_idcs = tf.reshape(tf.where(labels_flat == 1),[-1])

    out_ellipse_flat = tf.reshape(out_ellipse, [-1, 5])
    ellipse_targets_flat = tf.reshape(ellipse_targets, [-1, 5])

    out_ellipse_keep = tf.gather(out_ellipse_flat, pos_idcs, axis=0)
    ellipse_targets_keep = tf.gather(ellipse_targets_flat, pos_idcs, axis=0)

    out_tan = out_ellipse_keep[:, 4]
    out_angle = tf.atan(out_tan) * 180 / np.pi
    targets_tan = ellipse_targets_keep[:, 4]
    targets_angle = tf.atan(targets_tan) * 180 / np.pi

    err = tf.reduce_sum(tf.abs(out_angle - targets_angle)) / tf.cast(tf.shape(out_angle)[0], tf.float32)

    return err