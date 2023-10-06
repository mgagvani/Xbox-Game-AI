#!/usr/bin/env python


import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from tensorflow.keras.layers import Conv2D, Convolution2D, Conv2DTranspose
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import Sequence
from tensorflow.keras import optimizers, Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import Sample, load_data_from_samples

# Global variable
# OUT_SHAPE = 5
# OUT_SHAPE  = 16
OUT_SHAPE  = 1 # Change to number of variables used in output
INPUT_SHAPE = (Sample.IMG_H, Sample.IMG_W, Sample.IMG_D)


def customized_loss(y_true, y_pred, loss='euclidean'):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = K.mean(K.square((y_pred - y_true)), axis=-1) \
                    + K.sum(K.square(y_pred), axis=-1)/2 * L2_norm_cost
    # euclidean distance loss
    elif loss == 'euclidean':
        val = K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))
    return val

def create_model(keep_prob = 0.1):
    model = Sequential()

    # NVIDIA's model
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape= INPUT_SHAPE))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out)) 
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='softsign'))

    return model

def create_new_model(keep_prob = 0.7):
    model = Sequential()

    # NVIDIA's model
    # model.add(Lambda(lambda z: z / 127.5 - 1., input_shape=INPUT_SHAPE, output_shape=INPUT_SHAPE))
    # model.add(RandomBrightness(0.2, input_shape=INPUT_SHAPE))
    # model.add(RandomContrast(0.2, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='elu', input_shape= INPUT_SHAPE))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out)) 
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='tanh'))

    return model

def commaai_model(keep_prob):
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=INPUT_SHAPE,
            output_shape=INPUT_SHAPE))
  model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding="same"))
  model.add(ELU())
  model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
  model.add(ELU())
  model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(OUT_SHAPE))

  return model

def categorical_model(keep_prob=0.8):
    drop = 1 - keep_prob
    img_in = Input(shape=INPUT_SHAPE, name='img_in')
    x = Conv2D(24, 5, 2, padding='same')(img_in)  # Update this line
    x = Dropout(drop)(x)
    x = Conv2D(32, 5, 2, padding='same')(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, 5, 2, padding='same')(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, 3, 2, padding='same')(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, 3, 1, padding='same')(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu', name="dense_1")(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name="dense_2")(x)
    x = Dropout(drop)(x)
    # Categorical output of the angle into 15 bins
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    model = Model(inputs=[img_in], outputs=[angle_out], name='categorical')
    return model

def sequence_categorical_model(keep_prob=0.8, seq_len=5):
    drop = 1 - keep_prob
    seq_input_shape = (seq_len, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    seq_in = Input(shape=seq_input_shape, name='img_seq_in')
    x = TimeDistributed(Conv2D(24, 5, 2, padding='same'))(seq_in)
    x = TimeDistributed(Dropout(drop))(x)
    x = TimeDistributed(Conv2D(32, 5, 2, padding='same'))(x)
    x = TimeDistributed(Dropout(drop))(x)
    x = TimeDistributed(Conv2D(64, 5, 2, padding='same'))(x)
    x = TimeDistributed(Dropout(drop))(x)
    x = TimeDistributed(Conv2D(64, 3, 2, padding='same'))(x)
    x = TimeDistributed(Dropout(drop))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(100, activation='relu'))(x)
    x = TimeDistributed(Dropout(drop))(x)
    x = TimeDistributed(Dense(50, activation='relu'))(x)
    x = TimeDistributed(Dropout(drop))(x)
    # Categorical output of the angle into 15 bins
    x = TimeDistributed(Dense(15, activation='softmax'))(x)
    # Take the last time distributed output
    angle_out = Lambda(lambda x: x[:, -1], output_shape=(15,))(x)
    model = Model(inputs=[seq_in], outputs=[angle_out], name='seq_categorical')
    return model

class CustomDataGenerator(Sequence):
    def __init__(self, x_train, y_train, batch_size=32, seq_length=5):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_samples = x_train.shape[0]
        self.num_classes = y_train.shape[1]
        self.indexes = np.arange(self.seq_length, self.num_samples)

    def __len__(self):
        return int(np.ceil((self.num_samples - self.seq_length + 1) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        x_batch = []
        y_batch = []

        for i in batch_indexes:
            sequence_x = self.x_train[i - self.seq_length + 1:i + 1]
            x_batch.append(sequence_x)
            y_batch.append(self.y_train[i])

        return np.array(x_batch), np.array(y_batch)

def train_sequence_categorical_model(x_train, y_train, _model=sequence_categorical_model, batch_size=128, epochs=10):
    # Binning and one-hot encoding
    # Inputs are sequences of images (5 images each)
    print("Binning and one-hot encoding")
    num_bins = 15
    bin_edges = np.linspace(-1, 1, num_bins)
    y_binned = np.digitize(y_train, bin_edges)
    print(f"[DEBUG] bin_edges: {bin_edges}]")
    y = np.eye(num_bins)[y_binned - 1] 
    print("y_train shape: ", y_train.shape)
    # Reshape X to sequences of images
    seq_len = 5

    # get 20% of data for validation
    y_train = y[:int(len(y)*0.8)]
    y_val = y[int(len(y)*0.8):]

    x_train = x_train[:int(len(x_train)*0.8)]
    x_val = x_train[int(len(x_train)*0.8):]

    # use TF Dataset API to create generator (no duplicate data)
    train_generator= CustomDataGenerator(x_train, y_train, batch_size=batch_size, seq_length=seq_len)
    val_generator = CustomDataGenerator(x_val, y_val, batch_size=batch_size, seq_length=seq_len)
    # x_train = x_train_new
    # print("x_train shape: ", x_train.shape)

    # Train model
    model = _model()

    checkpoint = ModelCheckpoint("model_weights_seq_c1.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # default lr=0.001, make it smaller
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.00002))
    print(model.summary())
    model.fit(train_generator, epochs=epochs, shuffle=True, validation_data=val_generator, callbacks=callbacks_list)


def autoencoder_model():
    drop = 0.2
    img_in = Input(shape=INPUT_SHAPE, name='img_in')
    x = img_in
    x = Convolution2D(24, 5, strides=2, activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_6")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_7")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 1, strides=2, activation='relu', name="latent")(x)

    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_1")(x)
    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_2")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_3")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_4")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_5")(y)
    y = Conv2DTranspose(filters=1, kernel_size=3, strides=2, name="img_out")(y)

    x = Flatten(name='flattened')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    angle_out = Dense(15, activation='softmax', name='angle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out], name="latent")
    return model

def categorical_model_predict(loaded_model, input):
    pred = loaded_model.predict(input)
    # print(pred)
    # onehot --> -1 to 1 (15 bins)
    return (np.argmax(pred) / 7) - 1.0 # -1 to 1

def train_model(x_train, y_train, _model=create_model, batch_size=128, epochs=10):
    model = _model()

    checkpoint = ModelCheckpoint("model_weights_bal.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # model.compile(loss=customized_loss, optimizer="adam")
    # model.compile(loss="mean_squared_error", optimizer="adam")
    model.compile(loss="mean_squared_error", optimizer=optimizers.SGD(lr=0.1))
    print(model.summary())
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=callbacks_list)

def train_categorical_model(x_train, y_train, _model=categorical_model, batch_size=128, epochs=10):
    # Binning and one-hot encoding
    print("Binning and one-hot encoding")
    num_bins = 15
    # bin_edges = np.linspace(-1, 1, num_bins + 1)
    # y_binned = np.digitize(y_train, bin_edges)
    # y_train = np.eye(num_bins)[y_binned - 1] 
    # print("y_train shape: ", y_train.shape)
    y_train = tf.keras.utils.to_categorical(y_train, num_bins)

    model = _model()

    checkpoint = ModelCheckpoint("model_weights_c2a.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir="logs_C/", histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tensorboard]

    # default lr=0.001, make it smaller
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.00002))
    print(model.summary())
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=callbacks_list)

def train_autoencoder_model(x_train, y_train, _model=autoencoder_model, batch_size=128, epochs=10):
    print("Binning and one-hot encoding")
    num_bins = 15
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    y_binned = np.digitize(y_train, bin_edges)
    y_train = np.eye(num_bins)[y_binned - 1] 
    print("y_train shape: ", y_train.shape)

    model = _model()

    checkpoint = ModelCheckpoint("model_weights_A0.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # default lr=0.001, make it smaller
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.00002))
    print(model.summary())
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=callbacks_list)



if __name__ == '__main__':
    #Set GPU options
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    # model summary picture
    # tf.keras.utils.plot_model(sequence_categorical_model(), to_file='model.png', show_shapes=True)
    # exit()


    # Load Training Data
    print("loading training data")
    # x_train = np.load("data/x_sbal.npy")
    # y_train = np.load("data/y_sbal.npy")
    # samples = ['samples/forza4003', 'samples/forza4004', 'samples/forza4005']
    samples = ['samples/forza4003']
    x_train, y_train = load_data_from_samples(samples, debug=False, augment=True)

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 200
    batch_size = 256  

    # Train model
    # train_model(x_train, y_train, _model=create_model, batch_size=batch_size, epochs=epochs)
    train_categorical_model(x_train, y_train, _model=categorical_model, batch_size=batch_size, epochs=epochs)
    # train_autoencoder_model(x_train, y_train, _model=autoencoder_model, batch_size=batch_size, epochs=epochs)
    # train_sequence_categorical_model(x_train, y_train, _model=sequence_categorical_model, batch_size=batch_size, epochs=epochs)
