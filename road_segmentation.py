
"""## Using a TensorFlow version of the OpenVINO Road Segmentation Model to speed up inferencing
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = tf.saved_model.load("model/")

infer = model.signatures["serving_default"]

image = cv2.imread("./samples/forza8/img_56.jpg")
W = 896
H = 512
resized_image = cv2.resize(image, (W, H))
output = infer(tf.convert_to_tensor(resized_image.astype(np.float32)))
plt.imshow(output)
plt.show()
"""
# https://github.com/PINTO0309/PINTO_model_zoo/blob/main/136_road-segmentation-adas-0001/demo/demo_road-segmentation-adas-0001_tflite.py
# Currently a work in progress
# PRs are welcomed!
