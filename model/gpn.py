import tensorflow as tf
from model.base.vgg16 import VGG
from model.ellipse_target_layer import EllipseTargetLayer
from model.ellipse_proposal_layer import EllipseProposalLayer
from model.loss_layer import LossCls, LossEllipseSL1, LossEllipseKLD


class GPN(tf.keras.Model):

    def __init__(self, cfg):
        super(GPN, self).__init__()
        self.num_anchors = len(cfg['ANCHOR_SCALES'])
        self.ellipse_target = EllipseTargetLayer(cfg)
        self.ellipse_proposal = EllipseProposalLayer(cfg)

        if cfg['base_model'] == 'vgg16':
            self.base_model = VGG(cfg['pretrained'])
        else:
            raise Exception(
                'base model : {} not supported...'.format(cfg['base_model']))

        self.loss_cls = LossCls()

        if cfg['ELLIPSE_LOSS'] == 'KLD':
            self.loss_ellipse = LossEllipseKLD(cfg)
        elif cfg['ELLIPSE_LOSS'] == 'SL1':
            self.loss_ellipse = LossEllipseSL1()
        else:
            raise Exception(
                'ELLIPSE_LOSS : {} not supported...'.format(
                    cfg['ELLIPSE_LOSS']))

        self.conv_gpn = tf.keras.layers.Conv2D(
            filters=512, kernel_size=3, strides=1, padding='same', use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros()
        ) 
        self.relu_gpn = tf.keras.layers.ReLU()
        self.conv_cls = tf.keras.layers.Conv2D(
            filters=self.num_anchors * 2, kernel_size=1, strides=1, padding='valid', use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros()
        ) 
        self.conv_ellipse = tf.keras.layers.Conv2D(
            filters=self.num_anchors * 5, kernel_size=1, strides=1, padding='valid', use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros()
        ) 

    def call(self, img):
        base_feat = self.base_model(img)

        x = self.conv_gpn(base_feat)
        x = self.relu_gpn(x)
        #print(tf.shape(x))
        batch_size = tf.shape(x)[0]
        feat_height = tf.shape(x)[1]
        feat_width = tf.shape(x)[2] 

        out_cls = self.conv_cls(x)
        out_ellipse = self.conv_ellipse(x)

        
        out_cls = tf.reshape(tf.transpose(out_cls, [0, 2, 3, 1]), [batch_size, feat_height, feat_width, self.num_anchors, 2])

        out_ellipse = tf.reshape(tf.transpose(out_ellipse, [0, 2, 3, 1]), [batch_size, feat_height, feat_width, self.num_anchors, 5])

        return (out_cls, out_ellipse)
