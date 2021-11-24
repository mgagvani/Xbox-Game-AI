# Categorical Train - with EfficientNet Transfer Learning
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import efficientnet.keras as efn

from tensorflow.keras.callbacks import ModelCheckpoint
from utils import Sample

INPUT_SHAPE = (Sample.IMG_H, Sample.IMG_W, Sample.IMG_D)
NUM_CLASSES = 8

def create_model(keep_prob = 0.6):
    data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
    ]
    )

    base_model = keras.applications.EfficientNetB0(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=INPUT_SHAPE,
    include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    # NOTE Not doing this because ... 
    for layer in base_model.layers:
        layer.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = data_augmentation(inputs)  # Apply random data augmentation

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(NUM_CLASSES * 4, activation = "relu")(x)
    x = keras.layers.Dropout(1-keep_prob)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(NUM_CLASSES, activation = "sigmoid")(x)
    x = keras.layers.Dropout(1-keep_prob)(x)  # Regularize with dropout
    model = keras.Model(inputs, outputs)

    # print(model.summary()) 
    return model

def create_model_2(keep_prob = 0.3):
    model = efn.EfficientNetB3(input_shape = INPUT_SHAPE, include_top = False, weights = 'imagenet')
    for layer in model.layers:
        layer.trainable = False
    x = model.output

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(1-keep_prob)(x)

    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Output layer
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model_final = Model(inputs = model.input, outputs = predictions)

    print(model_final.summary())
    # model_final = keras.Model(base_model.input, outputs)

    # print(x.summary())
    
    return model_final
        
    


if __name__ == "__main__":
    #Set GPU options
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    print("loading training data")
    x_train = np.load("data/X_a1c.npy")  
    y_train = np.load("data/y_a1c.npy")
    print(x_train.shape[0], 'train samples')
    # Training loop variables
    epochs = 100
    batch_size = 50

    # model = create_model()
    model = create_model_2()

    checkpoint = ModelCheckpoint("model_weights_a1c2.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.compile(
    optimizer=keras.optimizers.Adam(),
    # loss=keras.losses.BinaryCrossentropy(from_logits=True),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    )
    model.build(x_train.shape)
    print(model.summary()) # NOTE we are not fine tuning the model TODO have to do it
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=callbacks_list)
