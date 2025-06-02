1) This repository contains the code for the TensorFlow implementation of the results from the paper:
Li, Yi. "Detecting lesion bounding ellipses with gaussian proposal networks." Machine Learning in Medical Imaging: 10th International Workshop, MLMI 2019, Held in Conjunction with MICCAI 2019, Shenzhen, China, October 13, 2019, Proceedings 10. Springer International Publishing, 2019.

2) Prerequisites:
Python 3.10.16
The rest of the libraries used are listed in requirements.txt
These can be installed using the command - 
pip install -r requirements.txt -i https://pypi.python.org/simple/

Please also download the pretrained VGG16 model weights file from the link (https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) and place it in the 'model/base' folder.

3) Data:
The entire dataset can be downloaded from the DeepLesion official release (https://nihcc.app.box.com/v/DeepLesion/). The raw CT images are located in 'Images_png', and the annotation is stored in 'DL_info.csv'. After downloading the whole dataset, please unzip all the *.zip files within the 'Images_png' folder.

4) Model:
The core idea of the Gaussian Proposal Network is modelling the lesion bounding ellipses as 2D Gaussian distributions on the image plane, and using KL-divergence loss for bounding ellipse localization.

5) Training:
Please train the model with the following command - 

python ./bin/train.py /CFG_PATH/cfg.json /SAVE_PATH/

 - where /CFG_PATH/ is the path to the config file in .json format, and /SAVE_PATH/ is where you want to save your model. Four config files are provided in the 'configs' folder, the following one is for gpn-5anchor -  
{
 "DATAPATH": "/DEEPLESION_PATH/",
 "MAX_SIZE": 512,
 "RPN_FEAT_STRIDE": 8,
 "ANCHOR_SCALES": [2, 3, 4, 6, 12],
 "ANCHOR_RATIOS": [1],
 "NORM_SPACING": 0.8,
 "SLICE_INTV": 2,
 "HU_MIN": -1024,
 "HU_MAX": 3071,
 "PIXEL_MEANS": 50,
 "MAX_NUM_GT_BOXES": 3,
 "TRAIN.RPN_BATCHSIZE": 32,
 "TRAIN.IMS_PER_BATCH": 2,
 "TRAIN.RPN_FG_FRACTION": 0.5,
 "TRAIN.RPN_POSITIVE_OVERLAP": 0.7,
 "TRAIN.RPN_NEGATIVE_OVERLAP": 0.3,
 "TRAIN.FROC_EVERY": 10,
 "TEST.IMS_PER_BATCH": 8,
 "TEST.RPN_NMS_THRESH": 0.3,
 "TEST.RPN_PRE_NMS_TOP_N": 6000,
 "TEST.RPN_POST_NMS_TOP_N": 300,
 "TEST.RPN_MIN_SIZE": 8,
 "TEST.FROC_OVERLAP": 0.5,
 "USE_GPU_NMS": true,
 "ELLIPSE_PAD": 5,
 "ELLIPSE_LOSS": "KLD",
 "base_model": "vgg16",
 "pretrained": true,
 "log_every": 100,
 "epoch": 20, 
 "lr": 0.001,
 "lr_factor": 0.1,
 "lr_epoch": 10,
 "momentum": 0.9,
 "grad_norm": 10.0,
 "weight_decay": 0.0005
}
Please modify /DEEPLESION_PATH/ to your path of the downloaded DeepLesion dataset. 'train.py' will generate a 'train' checkpoint, which is the most recently saved model, and a 'best' checkpoint, which is the model with the best validation FROC.


6) Testing:
We can evaluate the average FROC score of lesion localization by the command- 

python ./bin/test.py  /SAVE_PATH/

 - where /SAVE_PATH/ is where you saved your model. It will use the best checkpoint saved to compute the average FROC score on the official test split of the DeepLesion dataset.