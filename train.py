#!/usr/bin/env python


from random import Random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
    
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from tensorflow.keras.layers import Conv2D, Convolution2D
from tensorflow.keras import optimizers, Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import Sample

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
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    y_binned = np.digitize(y_train, bin_edges)
    y_train = np.eye(num_bins)[y_binned - 1] 
    print("y_train shape: ", y_train.shape)

    model = _model()

    checkpoint = ModelCheckpoint("model_weights_bal_cat.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    print(model.summary())
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=callbacks_list)

def load_data_from_samples(paths):
    # for each path, load y data from data.csv
    # 1st column is picture path, 2nd column is steering angle

    # determine number of samples
    num_samples = 0
    for path in paths:
        with open(path + "/data.csv") as f:
            num_samples += sum(1 for _line in f)
    
    # initialize x and y arrays
    x = np.empty((num_samples, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32)
    y = np.empty((num_samples), dtype=np.float32)

    # load data from each path
    i = 0
    for path in paths:
        with open(path + "/data.csv") as f:
            for line in f:
                tokens = line.split(",")
                # print(f"[DEBUG] {path + '/' + tokens[0]}")
                img = cv2.imread(tokens[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # resize image
                img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
                # if i % 500 == 0:
                #     plt.imshow(img)
                #     plt.title(tokens[1]+" "+str(i))
                #     plt.show()
                img = img.astype(np.float32)
                img = img / 127.5 - 1.0
                x[i] = img
                y[i] = float(tokens[1])
                print(f"sample {i} of {num_samples}", end="\r")
                i += 1
    
    return x, y

if __name__ == '__main__':
    #Set GPU options
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    
    # Load Training Data
    print("loading training data")
    # x_train = np.load("data/x_sbal.npy")
    # y_train = np.load("data/y_sbal.npy")
    x_train, y_train = load_data_from_samples(["samples/forza9",])

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 200
    batch_size = 128  

    # Train model
    # train_model(x_train, y_train, _model=create_model, batch_size=batch_size, epochs=epochs)
    train_categorical_model(x_train, y_train, _model=categorical_model, batch_size=batch_size, epochs=epochs)

    