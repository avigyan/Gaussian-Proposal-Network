import sys
import os
import argparse
import logging
import json
import time

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

tf.random.set_seed(0)

import model.utils as utils  # noqa
import model.functional as functional  # noqa
from data.deep_lesion_dataset import DeepLesionDataset  # noqa
from model.gpn import GPN  # noqa
from model.bbox_transform import bbox_overlaps  # noqa
from model.ellipse_transform import ellipse_overlaps  # noqa


parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=1, type=int, help='Number of'
                    ' workers for each data loader, default 1')
parser.add_argument('--iou_thred', default=0.5, type=float, help='IoU'
                    ' threshold, default 0.5')
parser.add_argument('--fps_img', default='0.5,1,2,4,8,16', type=str,
                    help='False positives per image, default 0.5,1,2,4,8,16')


def run(args):
    with open(os.path.join(args.save_path, 'cfg.json')) as f:
        cfg = json.load(f)

    model = GPN(cfg)
    ckpt_path = os.path.join(args.save_path, 'best.ckpt')
    model.load_weights(ckpt_path)
    model.trainable = False

    dataloader, steps = DeepLesionDataset(cfg, '3').get_tf_dataset(
        batch_size=cfg['TEST.IMS_PER_BATCH'], shuffle=False)

    #steps = len(dataloader)
    print(steps)
    time_now = time.time()
    loss_sum = 0
    loss_cls_sum = 0
    loss_ellipse_sum = 0
    n_proposals_sum = 0
    acc_pos_sum = 0
    acc_neg_sum = 0
    angle_err_sum = 0
    n_imgs = 0
    n_gt_boxes = 0
    froc_data_sum = np.zeros((steps, cfg['TEST.IMS_PER_BATCH'],
                              cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
    im_info = (cfg['MAX_SIZE'], cfg['MAX_SIZE'])

    for step, (img, gt_boxes, gt_ellipses) in enumerate(dataloader):
        gt_boxes = tf.convert_to_tensor(gt_boxes)
        gt_ellipses = tf.convert_to_tensor(gt_ellipses)

        labels, bbox_targets, ellipse_targets = model.ellipse_target(
            gt_boxes, gt_ellipses)

        out_cls, out_ellipse = model(tf.einsum("ijkl->iklj...", img), training=False)  # just  img
        loss_cls = model.loss_cls(out_cls, labels)
        loss_ellipse = model.loss_ellipse(out_ellipse, labels, ellipse_targets)
        loss = loss_cls + loss_ellipse
        acc_pos, acc_neg = functional.acc(out_cls, labels)
        n_proposals = functional.n_proposals(out_cls)
        angle_err = functional.angle_err(out_ellipse, labels, ellipse_targets)

        froc_data_batch = np.zeros((cfg['TEST.IMS_PER_BATCH'],
                                    cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
        for i in range(out_cls.shape[0]):
            # Final proposals and scores after NMS for each image
            boxes, ellipses, scores = model.ellipse_proposal(out_cls[i],
                                                             out_ellipse[i])
            # Keep non-padded gt_boxes/gt_ellipses
            keep = tf.reshape(tf.where(tf.reduce_sum(tf.cast(tf.greater(gt_ellipses[i], 0),dtype = tf.int64), axis=1) > 0),[-1])#[:, 0]
            overlaps = ellipse_overlaps(ellipses, tf.gather(gt_ellipses[i], keep), im_info)
            overlaps_max = tf.reduce_max(overlaps, axis=1)
            idcs_max = tf.argmax(overlaps, axis=1)
            n_ = scores.shape[0]
            n_imgs += 1
            n_gt_boxes += keep.shape[0]

            froc_data_batch[i, :n_, 0] = scores.numpy()
            froc_data_batch[i, :n_, 1] = overlaps_max.numpy()
            froc_data_batch[i, :n_, 2] = idcs_max.numpy()

        loss_sum += loss.numpy()
        loss_cls_sum += loss_cls.numpy()
        loss_ellipse_sum += loss_ellipse.numpy()
        n_proposals_sum += n_proposals.numpy()
        acc_pos_sum += acc_pos.numpy()
        acc_neg_sum += acc_neg.numpy()
        angle_err_sum += angle_err.numpy()
        froc_data_sum[step] = froc_data_batch

        if step % cfg['log_every'] == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg['log_every']
            loss_cls_sum /= cfg['log_every']
            loss_ellipse_sum /= cfg['log_every']
            n_proposals_sum = int(n_proposals_sum / cfg['log_every'])
            acc_pos_sum /= cfg['log_every']
            acc_neg_sum /= cfg['log_every']
            angle_err_sum /= cfg['log_every']

            logging.info(
                '{}, Test, Step : {}, Total Loss : {:.4f}, Cls Loss : {:.4f}, '
                'Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, Neg Acc : {:.3f}, '
                'Angle Err : {:.3f}, #Props/Img : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"), step, loss_sum,
                        loss_cls_sum, loss_ellipse_sum, acc_pos_sum,
                        acc_neg_sum, angle_err_sum, n_proposals_sum,
                        time_spent))

            loss_sum = 0
            loss_cls_sum = 0
            loss_ellipse_sum = 0
            n_proposals_sum = 0
            acc_pos_sum = 0.0
            acc_neg_sum = 0.0
            angle_err_sum = 0

    fps_img = list(map(float, args.fps_img.split(',')))
    FROC, sens = utils.froc(
        froc_data_sum.reshape((-1, cfg['TEST.RPN_POST_NMS_TOP_N'], 3)),
        n_imgs, n_gt_boxes, iou_thred=args.iou_thred, fps_img=fps_img)
    sens_str = '\t'.join(list(map(lambda x: '{:.3f}'.format(x), sens)))
    fps_img_str = '\t'.join(list(map(lambda x: '{:.2f}'.format(x), fps_img)))

    print('*' * 10 + 'False/Image' + '*' * 10)
    print(fps_img_str)
    print('*' * 10 + 'Sensitivity' + '*' * 10)
    print(sens_str)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()