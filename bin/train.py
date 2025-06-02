import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import tensorflow as tf

tf.print('Using TensorFlow version:', tf.__version__)
# Append parent directory for local imports (adjust if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')


import model.utils as utils  # noqa
import model.functional as functional  # noqa
from data.deep_lesion_dataset import DeepLesionDataset  # noqa
from model.gpn import GPN  # noqa
from model.bbox_transform import bbox_overlaps  # noqa
from model.ellipse_transform import ellipse_overlaps  # noqa

tf.random.set_seed(0)

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=1, type=int, help='Number of'
                    ' workers for each data loader, default 1')
parser.add_argument('--resume', default=0, type=int, help='If resume from'
                    ' previous run, default 0')

def clip_gradients(gradients, grad_norm):
    clipped_grads, _ = tf.clip_by_global_norm(gradients, grad_norm)
    return clipped_grads


def train_epoch(summary, writer, cfg, model, optimizer, train_dataset, len_train_dataset): 
    time_now = time.time()
    loss_sum = 0
    acc_pos_sum = 0
    acc_neg_sum = 0
    angle_err_sum = 0
    loss_cls_sum = 0
    loss_ellipse_sum = 0
    n_proposals_sum = 0
    n_imgs = 0
    n_gt_boxes = 0
    froc_idx = 0
    steps = len_train_dataset
    
    log_every = cfg['log_every']
    froc_every = cfg['TRAIN.FROC_EVERY']
    batch_size = cfg['TRAIN.IMS_PER_BATCH']
    froc_data_sum = np.zeros((log_every // froc_every, batch_size, cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
    im_info = (cfg['MAX_SIZE'], cfg['MAX_SIZE'])

    for step, (img, gt_boxes, gt_ellipses) in enumerate(train_dataset): 
        with tf.GradientTape() as tape:
            labels, bbox_targets, ellipse_targets = model.ellipse_target(gt_boxes, gt_ellipses) 
            out_cls, out_ellipse = model(tf.einsum("ijkl->iklj...", img), training=True)
            loss_cls = model.loss_cls(out_cls, labels)
            loss_ellipse = model.loss_ellipse(out_ellipse, labels, ellipse_targets)
            loss = loss_cls + loss_ellipse

        grads = tape.gradient(loss, model.trainable_variables)
        grads = clip_gradients(grads, cfg['grad_norm'])
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #tf.print('labels',type(labels),labels.shape,'ellipse_targets',type(ellipse_targets),ellipse_targets.shape) ###
        acc_pos, acc_neg = functional.acc(out_cls, labels)
        n_proposals = functional.n_proposals(out_cls)
        angle_err = functional.angle_err(out_ellipse, labels, ellipse_targets)

        loss_sum += loss.numpy() 
        loss_cls_sum += loss_cls.numpy()
        loss_ellipse_sum += loss_ellipse.numpy()
        acc_pos_sum += acc_pos.numpy()
        acc_neg_sum += acc_neg.numpy()
        angle_err_sum += angle_err.numpy()
        n_proposals_sum += n_proposals.numpy()

        summary['step'] += 1
        #tf.print('out_cls',type(out_cls),out_cls.shape,'out_ellipse',type(out_ellipse),out_ellipse.shape) ###
        if summary['step'] % froc_every == 0:
            froc_data_batch = np.zeros((batch_size, cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
            for i in range(out_cls.shape[0]):
                boxes, ellipses, scores = model.ellipse_proposal(out_cls[i], out_ellipse[i])
                #tf.print('ellipses',type(ellipses),ellipses.shape,'scores',type(scores),scores.shape) ###
                keep = tf.reshape(tf.where(tf.reduce_sum(tf.cast(tf.greater(gt_ellipses[i], 0),dtype = tf.int64), axis=1) > 0),[-1])#[:, 0]
                overlaps = ellipse_overlaps(ellipses, tf.gather(gt_ellipses[i], keep), im_info)
                #tf.print('overlaps',type(overlaps),overlaps.shape) ###
                overlaps_max = tf.reduce_max(overlaps, axis=1)
                idcs_max = tf.argmax(overlaps, axis=1)
                n_ = scores.shape[0]

                froc_data_batch[i, :n_, 0] = scores.numpy() 
                froc_data_batch[i, :n_, 1] = overlaps_max.numpy() 
                froc_data_batch[i, :n_, 2] = idcs_max.numpy() 

                n_imgs += 1
                n_gt_boxes += keep.shape[0]
                #tf.print('keep',type(keep),keep.shape) ###

            froc_data_sum[froc_idx] = froc_data_batch
            froc_idx += 1

        if summary['step'] % log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= log_every 
            loss_cls_sum /= log_every 
            loss_ellipse_sum /= log_every 
            acc_pos_sum /= log_every 
            acc_neg_sum /= log_every 
            angle_err_sum /= log_every 
            n_proposals_sum = int(n_proposals_sum / log_every) 

            #tf.print('froc_data_sum',type(froc_data_sum),froc_data_sum.shape) ###
            #tf.print('n_imgs',type(n_imgs),n_imgs,'n_gt_boxes',type(n_gt_boxes),n_gt_boxes) ###
            FROC, sens = utils.froc(
                froc_data_sum.reshape((-1, cfg['TEST.RPN_POST_NMS_TOP_N'], 3)),
                n_imgs, n_gt_boxes, iou_thred=cfg['TEST.FROC_OVERLAP']
            )
            
            sens_str = ' '.join(list(map(lambda x: '{:.3f}'.format(x), sens)))
            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Total Loss : {:.4f}, '
                'Cls Loss : {:.4f}, Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, '
                'Neg Acc : {:.3f}, Angle Err : {:.3f}, FROC : {:.3f}, '
                'Sens : {}, #Props/Img : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_sum, 
                        loss_cls_sum, loss_ellipse_sum, acc_pos_sum, 
                        acc_neg_sum, angle_err_sum, FROC, sens_str, 
                        n_proposals_sum, time_spent)) 
            #tf.print(step)
            with writer.as_default():
                tf.summary.scalar('train/total_loss', loss_sum, step=summary['step']) 
                tf.summary.scalar('train/loss_cls', loss_cls_sum, step=summary['step']) 
                tf.summary.scalar('train/loss_ellipse', loss_ellipse_sum, step=summary['step']) 
                tf.summary.scalar('train/acc_pos', acc_pos_sum, step=summary['step']) 
                tf.summary.scalar('train/acc_neg', acc_neg_sum, step=summary['step']) 
                tf.summary.scalar('train/angle_err', angle_err_sum, step=summary['step']) 
                tf.summary.scalar('train/n_proposals', n_proposals_sum, step=summary['step']) 
                tf.summary.scalar('train/FROC', FROC, step=summary['step'])

            loss_sum = 0
            loss_cls_sum = 0
            loss_ellipse_sum = 0
            acc_pos_sum = 0
            acc_neg_sum = 0
            angle_err_sum = 0
            n_proposals_sum = 0
            n_imgs = 0
            n_gt_boxes = 0
            froc_idx = 0
            froc_data_sum = np.zeros((log_every // froc_every, batch_size, cfg['TEST.RPN_POST_NMS_TOP_N'], 3))

    summary['epoch'] += 1
    return summary

def valid_epoch(summary, cfg, model, valid_dataset, len_valid_dataset): 
    loss_sum = 0
    acc_pos_sum = 0
    acc_neg_sum = 0
    angle_err_sum = 0
    loss_cls_sum = 0
    loss_ellipse_sum = 0
    n_proposals_sum = 0
    n_imgs = 0
    n_gt_boxes = 0
    steps = len_valid_dataset 
    
    froc_data_sum = np.zeros((steps, cfg['TEST.IMS_PER_BATCH'], cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
    im_info = (cfg['MAX_SIZE'], cfg['MAX_SIZE'])

    for step, (img, gt_boxes, gt_ellipses) in enumerate(valid_dataset):
        labels, bbox_targets, ellipse_targets = model.ellipse_target(gt_boxes, gt_ellipses) # cuda???
        out_cls, out_ellipse = model(tf.einsum("ijkl->iklj...", img), training=False)
        loss_cls = model.loss_cls(out_cls, labels)
        loss_ellipse = model.loss_ellipse(out_ellipse, labels, ellipse_targets)
        loss = loss_cls + loss_ellipse

        #tf.print('labels',type(labels),labels.shape,'ellipse_targets',type(ellipse_targets),ellipse_targets.shape) ###
        acc_pos, acc_neg = functional.acc(out_cls, labels)
        n_proposals = functional.n_proposals(out_cls)
        angle_err = functional.angle_err(out_ellipse, labels, ellipse_targets)

        #tf.print('out_cls',type(out_cls),out_cls.shape,'out_ellipse',type(out_ellipse),out_ellipse.shape) ###
        froc_data_batch = np.zeros((cfg['TEST.IMS_PER_BATCH'], cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
        for i in range(out_cls.shape[0]):
            boxes, ellipses, scores = model.ellipse_proposal(out_cls[i], out_ellipse[i])
            #tf.print('ellipses',type(ellipses),ellipses.shape,'scores',type(scores),scores.shape) ###
            keep = tf.reshape(tf.where(tf.reduce_sum(tf.cast(tf.greater(gt_ellipses[i], 0), dtype = tf.int64), axis=1) > 0),[-1])#[:, 0]
            overlaps = ellipse_overlaps(ellipses, tf.gather(gt_ellipses[i], keep), im_info)
            overlaps_max = tf.reduce_max(overlaps, axis=1)
            idcs_max = tf.argmax(overlaps, axis=1)
            n_ = scores.shape[0]

            froc_data_batch[i, :n_, 0] = scores.numpy()
            froc_data_batch[i, :n_, 1] = overlaps_max.numpy() 
            froc_data_batch[i, :n_, 2] = idcs_max.numpy() 
            n_imgs += 1
            n_gt_boxes += keep.shape[0]

        
        loss_sum += loss.numpy() 
        loss_cls_sum += loss_cls.numpy()
        loss_ellipse_sum += loss_ellipse.numpy()
        acc_pos_sum += acc_pos.numpy()
        acc_neg_sum += acc_neg.numpy()
        angle_err_sum += angle_err.numpy()
        n_proposals_sum += n_proposals.numpy()
        froc_data_sum[step] = froc_data_batch

    FROC, sens = utils.froc(froc_data_sum.reshape((-1, cfg['TEST.RPN_POST_NMS_TOP_N'], 3)), n_imgs, n_gt_boxes, iou_thred=cfg['TEST.FROC_OVERLAP'])
    summary.update({
        'loss': loss_sum / steps,
        'loss_cls': loss_cls_sum / steps,
        'loss_ellipse': loss_ellipse_sum / steps,
        'acc_pos': acc_pos_sum / steps,
        'acc_neg': acc_neg_sum / steps,
        'angle_err': angle_err_sum / steps,
        'n_proposals': int(n_proposals_sum / steps),
        'FROC': FROC,
        'sens_str': ' '.join(map(lambda x: f'{x:.3f}', sens))
    })
    return summary

def run(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    os.makedirs(args.save_path, exist_ok=True)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    model = GPN(cfg)
    optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['lr'], momentum=cfg['momentum'], decay = cfg['weight_decay']) 
    train_dataset, len_train_dataset = DeepLesionDataset(cfg, '1').get_tf_dataset(
        batch_size=cfg['TRAIN.IMS_PER_BATCH'], shuffle=True)
    
    
    valid_dataset, len_valid_dataset = DeepLesionDataset(cfg, '2').get_tf_dataset(
        batch_size=cfg['TEST.IMS_PER_BATCH'], shuffle=False)

    summary_writer = tf.summary.create_file_writer(args.save_path)
    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'loss_cls': float('inf'), 'loss_ellipse': float('inf')}
    FROC_valid_best = 0
    epoch_start = 0

    checkpoint_path = os.path.join(args.save_path, 'train.ckpt') 
    if args.resume and os.path.exists(checkpoint_path + '.index'): 
        model.load_weights(checkpoint_path)
        with open(os.path.join(args.save_path, 'resume.json')) as f:
            resume_data = json.load(f)
        summary_train.update(resume_data['summary_train'])
        FROC_valid_best = resume_data['FROC_valid_best']
        epoch_start = summary_train['epoch']

    for epoch in range(epoch_start, cfg['epoch']):
        lr = utils.lr_schedule(cfg['lr'], cfg['lr_factor'], summary_train['epoch'], cfg['lr_epoch'])
        optimizer.learning_rate.assign(lr)

        summary_train = train_epoch(summary_train, summary_writer, cfg, model, optimizer, train_dataset, len_train_dataset)
        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, cfg, model, valid_dataset, len_valid_dataset)
        time_spent = time.time() - time_now

        logging.info(
            '{}, Valid, Epoch : {}, Step : {}, Total Loss : {:.4f}, '
            'Cls Loss : {:.4f}, Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, '
            'Neg Acc : {:.3f}, Angle Err : {:.3f}, FROC : {:.3f}, Sens : {}, '
            '#Props/Img : {}, Run Time : {:.2f} sec'
            .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['epoch'], summary_train['step'],
                    summary_valid['loss'], summary_valid['loss_cls'],
                    summary_valid['loss_ellipse'], summary_valid['acc_pos'],
                    summary_valid['acc_neg'], summary_valid['angle_err'],
                    summary_valid['FROC'], summary_valid['sens_str'],
                    summary_valid['n_proposals'], time_spent))

        with summary_writer.as_default():
            tf.summary.scalar('valid/total_loss', summary_valid['loss'], step=summary_train['step'])
            tf.summary.scalar('valid/loss_cls', summary_valid['loss_cls'], step=summary_train['step'])
            tf.summary.scalar('valid/loss_ellipse', summary_valid['loss_ellipse'], step=summary_train['step'])
            tf.summary.scalar('valid/n_proposals', summary_valid['n_proposals'], step=summary_train['step'])
            tf.summary.scalar('valid/acc_pos', summary_valid['acc_pos'], step=summary_train['step'])
            tf.summary.scalar('valid/acc_neg', summary_valid['acc_neg'], step=summary_train['step'])
            tf.summary.scalar('valid/angle_err', summary_valid['angle_err'], step=summary_train['step'])
            tf.summary.scalar('valid/FROC', summary_valid['FROC'], step=summary_train['step'])

        tf.print("################# summary_valid['FROC'] : ", summary_valid['FROC'])
        tf.print("################# FROC_valid_best : ", FROC_valid_best)
        if summary_valid['FROC'] > FROC_valid_best:
            FROC_valid_best = summary_valid['FROC']
            model.save_weights(os.path.join(args.save_path, 'best.ckpt'))
            
            logging.info(
                '{}, Best, Epoch : {}, Step : {}, Total Loss : {:.4f}, '
                'Cls Loss : {:.4f}, Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, '
                'Neg Acc : {:.3f}, Angle Err : {:.3f}, FROC : {:.3f}, '
                'Sens : {}, #Props/Img : {}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary_train['epoch'], summary_train['step'],
                        summary_valid['loss'], summary_valid['loss_cls'],
                        summary_valid['loss_ellipse'],
                        summary_valid['acc_pos'], summary_valid['acc_neg'],
                        summary_valid['angle_err'], summary_valid['FROC'],
                        summary_valid['sens_str'],
                        summary_valid['n_proposals']
                        ))

        model.save_weights(checkpoint_path)
        with open(os.path.join(args.save_path, 'resume.json'), 'w') as f:
            json.dump({
                'summary_train': summary_train,
                'FROC_valid_best': FROC_valid_best
            }, f)

    summary_writer.close()

def main(): # ok
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
