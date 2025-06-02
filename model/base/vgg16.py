import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import math
import numpy as np
import torch  # Required for loading PyTorch weights
import requests
from tensorflow.keras.applications import VGG16
model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512]
}

layerNames = ["block1_conv1","block1_conv2",#"block1_pool",
"block2_conv1","block2_conv2",#"block2_pool",
"block3_conv1","block3_conv2","block3_conv3",#"block3_pool",
"block4_conv1","block4_conv2","block4_conv3",#"block4_pool",
"block5_conv1","block5_conv2","block5_conv3"]


def make_layers(cfg, batch_norm=False):  
    model_layers = []
    in_channels = 3  # TensorFlow infers this from input shape, used only to track filter count
    cnt = 0
    for v in cfg:
        if v == 'M':
            model_layers.append(layers.MaxPooling2D(pool_size=2, strides=2))
        else:
            conv2d = layers.Conv2D(
                filters=v,
                kernel_size=3,
                padding='same',
                activation=None,
                use_bias=True, #not batch_norm
                name = layerNames[cnt]
            )
            model_layers.append(conv2d)
            if batch_norm:
                model_layers.append(layers.BatchNormalization())
            model_layers.append(layers.ReLU())
            in_channels = v  # Not strictly needed in TF
            cnt += 1

    model = Sequential(model_layers)###  
    model.build((None, 512, 512, 3))###
    model.load_weights('./model/base/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True, skip_mismatch=True)
    #model.summary()
    return model 

class VGG(Model):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.withBatchNorm = False  # Set to True if you want to use Batch Normalization
        self.features = make_layers(cfg['D'],batch_norm=self.withBatchNorm)  # Create the VGG16 feature extractor
        self.pretrained = pretrained
        #self._initialize_weights()
        
        #if self.pretrained:
        #    self.load_pretrained()

    def call(self, inputs): 
        return self.features(inputs)

    def _initialize_weights(self):
        for layer in self.features.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.filters
                stddev = math.sqrt(2. / n)
                if layer.kernel is not None:
                    layer.kernel.assign(tf.random.normal(shape=layer.kernel.shape, mean=0.0, stddev=stddev))
                if layer.bias is not None:
                    layer.bias.assign(tf.zeros_like(layer.bias))

            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                if layer.gamma is not None:
                    layer.gamma.assign(tf.ones_like(layer.gamma))
                if layer.beta is not None:
                    layer.beta.assign(tf.zeros_like(layer.beta))

            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.kernel is not None:
                    layer.kernel.assign(tf.random.normal(shape=layer.kernel.shape, mean=0.0, stddev=0.01))
                if layer.bias is not None:
                    layer.bias.assign(tf.zeros_like(layer.bias))

        if self.pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        """Load pretrained weights from PyTorch VGG16"""
        # Download and load PyTorch state_dict
        state_dict = torch.hub.load_state_dict_from_url(model_urls['vgg16'])
        model = VGG16(weights='imagenet')
        model.load_weights('./model/base/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True, skip_mismatch=True)
        for layer in model.layers:
            print(layer.name)
        # Mapping between TensorFlow layers and PyTorch state_dict keys
        if self.withBatchNorm:
            layer_mapping = [                           # original index                       With BatchNorm      WO BatchNorm
                (0, 'features.0'),   # First Conv2D     # 0                                         0                   0
                (3, 'features.2'),    # Second Conv2D   # 1                                         3                   2   
                (7, 'features.5'),    # Third Conv2D    # 3                                         7                   5
                (10, 'features.7'),    # Fourth Conv2D   # 4                                        10                  7         
                (14, 'features.10'),   # Fifth Conv2D    # 6                                        14                  10            
                (17, 'features.12'),   # Sixth Conv2D    # 7                                        17                  12    
                (20, 'features.14'),   # Seventh Conv2D  # 8                                        20                  14            
                (24, 'features.17'),  # Eighth Conv2D   # 10                                        24                  17    
                (27, 'features.19'),  # Ninth Conv2D    # 11                                        27                  19    
                (30, 'features.21'),  # Tenth Conv2D    # 12                                        30                  21    
                (33, 'features.24'),  # Eleventh Conv2D (adjusted index)  # 13                      33                  23                                                          
                (36, 'features.26'),  # Twelfth Conv2D (adjusted index)   # 14                      36                  25    
                (39, 'features.28'),  # Thirteenth Conv2D (adjusted index) # 15                     39                  27    
            ]

        else:
            layer_mapping = [                           # original index                       With BatchNorm      WO BatchNorm
                (0, 'features.0'),   # First Conv2D     # 0                                         0                   0
                (2, 'features.2'),    # Second Conv2D   # 1                                         3                   2   
                (5, 'features.5'),    # Third Conv2D    # 3                                         7                   5
                (7,  'features.7'),    # Fourth Conv2D   # 4                                        10                  7         
                (10, 'features.10'),   # Fifth Conv2D    # 6                                        14                  10            
                (12, 'features.12'),   # Sixth Conv2D    # 7                                        17                  12    
                (14, 'features.14'),   # Seventh Conv2D  # 8                                        20                  14            
                (17, 'features.17'),  # Eighth Conv2D   # 10                                        24                  17    
                (19, 'features.19'),  # Ninth Conv2D    # 11                                        27                  19    
                (21, 'features.21'),  # Tenth Conv2D    # 12                                        30                  21    
                (23, 'features.24'),  # Eleventh Conv2D (adjusted index)  # 13                      33                  23                                                          
                (25, 'features.26'),  # Twelfth Conv2D (adjusted index)   # 14                      36                  25    
                (27, 'features.28'),  # Thirteenth Conv2D (adjusted index) # 15                     39                  27    
            ]
        for i,layer in enumerate(self.features.layers):
           print(i,layer.name)
        #print(self.features.layers)
        for tf_idx, pt_prefix in layer_mapping:
            # Get PyTorch weights and biases
            pt_weight = state_dict[f'{pt_prefix}.weight'].numpy()
            pt_bias = state_dict[f'{pt_prefix}.bias'].numpy()
            
            # Convert from PyTorch (OIHW) to TensorFlow (HWIO) format
            tf_weight = np.transpose(pt_weight, (2, 3, 1, 0))
            
            # Get corresponding TensorFlow layer
            tf_layer = self.features.layers[tf_idx]
            

            #if tf_idx == 0:
                # For the first layer, set input shape
            #    tf_layer.build((None, None, None, 3))
            #print(tf_idx)
            # Set weights
            tf_layer.set_weights([tf_weight, pt_bias])

# Example usage
if __name__ == "__main__":
    model = VGG(pretrained=True)
    model.build((None, 512, 512, 3)) 
    model.summary()