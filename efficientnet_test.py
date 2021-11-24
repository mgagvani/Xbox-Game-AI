import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image

# EfficientNet

import tensorflow
from tensorflow.keras.applications.imagenet_utils import decode_predictions as decode_efnet

from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize 
from efficientnet.tfkeras import preprocess_input as preprocess_efficientnet

# ResNetV2
from tensorflow.keras.applications.resnet_v2 import decode_predictions as decode_resnet

from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications import ResNet50V2

def load_data():
    image = imread("./samples/2021-05-02_17;54;46/img_191.png")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

    return image

def efficientnet_inference(image):
    model = EfficientNetB0(weights='imagenet')

    image_size = model.input_shape[1]
    print(image_size)
    x = center_crop_and_resize(image, image_size=image_size)
    x = preprocess_efficientnet(x)
    x = np.expand_dims(x, 0)

    y = model.predict(x)
    print(decode_efnet(y))

def resnet_inference(image):
    model = ResNet50V2(weights='imagenet')

    image_size = model.input_shape[1]
    print(image_size)
    x = np.resize(image,(224,224,3))
    x = preprocess_resnet(x)
    x = np.expand_dims(x, 0)

    y = model.predict(x)
    print(decode_resnet(y))


image = load_data()
print("efficientnet")
efficientnet_inference(image)
print("resnet")
resnet_inference(image)
