# This program tests whether you have a GPU that Tensorflow can access. 

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
print(gpus)

