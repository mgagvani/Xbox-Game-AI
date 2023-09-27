import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np


from utils import load_data_from_samples

def create_efficientnet_model():
    base_model = EfficientNetV2B1(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1, activation='relu')(x)

    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    return model

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    # load data
    paths = ["samples/forza4003", "samples/forza4004", "samples/forza4005"]
    x, y = load_data_from_samples(paths)

    # num_bins = 15
    # bin_edges = np.linspace(-1, 1, num_bins + 1)
    # y_binned = np.digitize(y_float, bin_edges)
    # y = np.eye(num_bins)[y_binned - 1]

    # create model
    model = create_efficientnet_model()
    model.load_weights("model_weights_E0.h5")
    print(model.summary())
    '''
    # compile model
    checkpoint0 = ModelCheckpoint("model_weights_E0.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard0 = TensorBoard(log_dir="logs_E/", histogram_freq=0, write_graph=True, write_images=True)
    model.compile(optimizer='adam', loss="mse", metrics=[])

    # train model
    model.fit(x, y, epochs=5, batch_size=64, validation_split=0.2, callbacks=[checkpoint0, tensorboard0])
    '''

    # fine-tune model
    checkpoint = ModelCheckpoint("model_weights_E0.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir="logs_E/", histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer='adam', loss='mse', metrics=[])
    model.fit(x, y, epochs=10, batch_size=8, validation_split=0.2, callbacks=callbacks_list)


