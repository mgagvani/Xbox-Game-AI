#!/usr/bin/env python

import sys
import gc
import glob
import time
import os
import random
from PIL import Image
import mss
import pandas as pd

try:
    # import cupy as np
    import numpy as np
except ImportError:
    print("Cupy not installed.")
    import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread

# comment these out when using WSL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import cv2
from inputs import get_gamepad
import math
import threading

class Screenshotter(object):
    def __init__(self):
        # import openvino_test
        self.sct = mss.mss()
        self.vec = None

        self._monitor_thread = threading.Thread(target=self._get_image, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        # self.ie, self.net, self.exec_net, self.output_layer_ir, self.input_layer_ir = openvino_test.start()
        
    def _get_image(self):
        while True:
            time.sleep(0.02)
            # Get raw pixels from the screen
            t1 = time.perf_counter()
            sct_img = self.sct.grab({  "top":Screenshot.OFFSET_Y,
                                    "left": Screenshot.OFFSET_X,
                                    "width": Screenshot.SRC_W,
                                    "height": Screenshot.SRC_H})
            # Create the Image
            # print(f'[DEBUG] Screenshot took {time.perf_counter() - t1} seconds')
            temp = np.array(Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX'))
            #Perform segmentations
            # temp = self.convert_to_segmented(temp)

            # DEBUG
            # import matplotlib.pyplot as plt

            # Resize
            self.vec = cv2.resize(temp, (Sample.IMG_W, Sample.IMG_H), interpolation=cv2.INTER_LINEAR_EXACT)
            self.vec = cv2.cvtColor(self.vec, cv2.COLOR_BGR2RGB)
            # Augmentations
            # vec = cv2.rectangle(img=vec.astype(np.uint8), pt1=(int(0),int(0)), pt2=(int(480), int(90)), color=[0, 0, 0], thickness=cv2.FILLED)
            # return vec

    def take_screenshot(self):
        if self.vec is None:
            time.sleep(0.1)
        # copy
        vec = self.vec.copy()
        return vec

    # def convert_to_segmented(self, img):
    #     return openvino_test.inference(img, self.ie, self.net, self.exec_net, self.output_layer_ir, self.input_layer_ir, True) 


def resize_image(img):
    im = resize(img, (Sample.IMG_H, Sample.IMG_W, Sample.IMG_D))
    im_arr = im.reshape((Sample.IMG_H, Sample.IMG_W, Sample.IMG_D))
    return im_arr


class Screenshot(object):
    SRC_W = 1920
    SRC_H = 1080
    # SRC_W = 300
    # SRC_H = 300
    SRC_D = 3

    OFFSET_X = 320 # because of ultrawide monitor
    OFFSET_Y = 0


    # OFFSET_X = 1920
    # OFFSET_Y = 780


class Sample(object):
    IMG_W = 240 # 480
    IMG_H = 135 # 270 
    # IMG_W = 300
    # IMG_H = 300
    IMG_D = 3


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self):
        L_X = self.LeftJoystickX
        L_Y = self.LeftJoystickY
        R_X = self.RightJoystickX
        R_Y = self.RightJoystickY
        LT = self.LeftTrigger
        RT = self.RightTrigger
        LB = self.LeftBumper
        RB = self.RightBumper
        A = self.A
        X = self.X
        Y = self.Y
        B = self.B
        LTh = self.LeftThumb
        RTh = self.RightThumb
        Back = self.Back
        Start = self.Start
        # dpad does not work
        DP_L = self.LeftDPad
        DP_R = self.RightDPad
        DP_U = self.UpDPad
        DP_D = self.DownDPad

        # return [L_X, L_Y, R_X, R_Y, RT]
        return [L_X, L_Y, R_X, R_Y, LT, RT, LB, RB, A, X, Y, B, LTh, RTh, Back, Start]
        # return [L_X, L_Y, R_X, R_Y, RT]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'BTN_WEST':
                    self.Y = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state


class Data(object):
    def __init__(self):
        self._X = np.load("data/X.npy")
        self._y = np.load("data/y.npy")
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._X.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


def load_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)) 
    return image_files, joystick_values

def load_mini_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5,6,9,10)) 
    return image_files, joystick_values

def load_categorical_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(17,)) 
    return image_files, joystick_values

def load_racing_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,5,6)) 
    return image_files, joystick_values

def load_steering_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,)) 
    return image_files, joystick_values

def load_imgs(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    return image_files

def load_balanced_sample(samples, col="LX", bias=0.2):
    """
    Samples: List of all CSV files to concat and balance
    Col: Column to balance by. By default, "LX"
    """
    cols = ["Name", "LX", "LY", "RX", "RY", "LT", "RT"]
    for i in range(10):
        cols.append(str(i))
    dataframes = [pd.read_csv(sample) for sample in samples]
    for f in dataframes:
        f.columns = cols
    concat = pd.concat(dataframes, axis=0, ignore_index=True)
    print(concat)
    concat.hist(column=["LX", "RT"], bins=200)
    plt.show()
    df = concat
    # find concat fraction to chop off (assume left and right are equal)
    fract = df[(df[col] > 0.1) & (df[col] < 1.0)].shape[0]/df.shape[0]
    fract = fract * bias
    new_df = df[(df[col] < -0.1) | (df[col] > 0.1) | (abs(df[col]) < 0.1).sample(frac=fract)]
    new_df.hist(column=["LX", "RT"], bins=200)
    print(new_df)
    plt.show()
    return new_df["Name"], new_df[col]

def ask_for_samples():
    from train import load_data_from_samples
    # ask for samples
    samples = eval(input("Enter sample paths to load: "))
    # load data
    return load_data_from_samples(samples, augment=False)

def plot_data(y_pth, predictions=False, model_pth=None, x_pth=None, categorical=False):
    categorical = True if categorical == "y" else False

    # check if model exists
    if os.path.exists(model_pth):
        print(f"Model {model_pth} found")
    else:
        raise FileNotFoundError(f"Model {model_pth} not found")
    
    # load data
    # X = np.load(x_pth)
    if input("Load data from samples? (y/n): ") == "y":
        print("Loading data from samples...")
        X, y = ask_for_samples()
    else:
        print("Loading data from NPY files...")
        y = np.load(y_pth)
        # load X data
        X = np.load(x_pth)

    if (not categorical) and input("Mediapipe? (y/n): ") == "y":
        mediapipe = True
    else:
        mediapipe = False

    # plot y data
    plt.plot(y)

    if categorical:
        print("Categorical data")
    else:
        print("Continuous data")

    # plot predictions
    if predictions and (not categorical) and (not mediapipe):
        from train import commaai_model, create_model, create_new_model
        from train_categorical import create_efficientnet_model
        # load model
        # model = create_model(keep_prob=1.0)
        model = create_efficientnet_model()
        model.load_weights(model_pth)
        # predict
        y_preds = []
        t0 = time.perf_counter()
        for i, x in enumerate(X):
            print(i, "/", len(X)-1, end="\r")
            y_pred = model.predict(np.expand_dims(x, axis=0), batch_size=1)[0]
            y_preds.append(y_pred)
        t1 = time.perf_counter()
        print("time per prediction:", (t1-t0)/len(X), "seconds")
        # plot
        plt.plot(y_preds)  
    elif predictions and categorical and (not mediapipe):
        from train import categorical_model, autoencoder_model, categorical_model_predict
        import tensorflow as tf
        # cuda memory growth
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        # load model
        model = categorical_model()
        # model = autoencoder_model()
        model.load_weights(model_pth)
        # predict
        y_preds = []
        t0 = time.perf_counter(); y_pred = 0
        for i, x in enumerate(X):
            print(i, "/", len(X)-1, y_pred, end="\r")
            y_pred = categorical_model_predict(model, np.expand_dims(x, axis=0))
            y_preds.append(y_pred)
        t1 = time.perf_counter()
        print("time per prediction:", (t1-t0)/len(X), "seconds")
        # plot
        plt.plot(y_preds)
    elif mediapipe:
        # load tflite model
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python.components import processors
        from mediapipe.tasks.python import vision
        print("Running mediapipe model...")

        # STEP 2: Create an ImageClassifier object.
        base_options = python.BaseOptions(model_asset_path=model_pth)
        options = vision.ImageClassifierOptions(
            base_options=base_options, max_results=4)
        classifier = vision.ImageClassifier.create_from_options(options)

        # image = mp.Image.create_from_file(image_name)
        # classification_result = classifier.classify(image)
        # predict
        y_preds = []
        t0 = time.perf_counter()
        for i, x in enumerate(X):
            print(i, "/", len(X)-1, end="\r")
            # predict

            # convert from float 1.0 to uint8 255
            x = (x*255).astype(np.uint8)
            # debug show image
            # plt.imshow(x)
            # plt.show()

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=x)
            classification_result = classifier.classify(image).classifications[0]
            # for i in classification_result.categories:
            #     print(i)
        
            # convert idx to float
            y_pred = float(classification_result.categories[0].index) / 7.0 - 1.0
            y_preds.append(y_pred)
        t1 = time.perf_counter()
        print("time per prediction:", (t1-t0)/len(X), "seconds")
        # plot
        plt.plot(y_preds)
    else:
        raise ValueError(f'Invalid arguments: predictions={predictions}, categorical={categorical}, mediapipe={mediapipe}')
    plt.show()

def seq_plotpreds(data_pths, model_pth, categorical=True, seq_len=5):
    # check if model exists
    if os.path.exists(model_pth):
        print(f"Model {model_pth} found")
    else:
        raise FileNotFoundError(f"Model {model_pth} not found")
    
    seq_len = int(seq_len)

    print(data_pths)
    x, y = load_data_from_samples(eval(data_pths), augment=False)

    # open model
    from train import sequence_categorical_model
    model = sequence_categorical_model(seq_len=seq_len)
    model.load_weights(model_pth)

    # predict
    y_preds = []
    t0 = time.perf_counter()
    for i in range(seq_len, len(x)):
        print(i, "/", len(x)-1, end="\r")
        # take last seq_len images
        y_pred = model.predict(np.expand_dims(x[i-seq_len:i], axis=0), batch_size=1)[0]

        # covert from idx to float
        y_pred = np.argmax(y_pred) / 7.0 - 1.0

        y_preds.append(y_pred)
    t1 = time.perf_counter()

    print("time per prediction:", (t1-t0)/len(x), "seconds")

    # plot
    plt.plot(y_preds)
    plt.plot(y[seq_len:])
    plt.show()

def show_pic(x_pth, idx=0):
    # load only one image from samples
    samples = eval(input("Enter sample paths to load: "))
    # do NOT load entire array - find the image path and load it
    image_files = []
    for sample in samples:
        print(sample, end="\r")
        x = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
        for a in x:
            image_files.append(a)
    print(f'[DEBUG] {len(image_files)} images found')
    # load image
    img = imread(image_files[idx])
    # show image
    plt.imshow(img)
    plt.show()

# training data viewer
def viewer(sample):
    image_files, joystick_values = load_sample(sample)

    plotData = []

    plt.ion()
    plt.figure('viewer', figsize=(16, 6))

    for i in range(len(image_files)):

        # joystick
        print(i, " ", joystick_values[i,:])

        # format data
        plotData.append( joystick_values[i,:] )
        if len(plotData) > 30:
            plotData.pop(0)
        x = np.asarray(plotData)

        # image (every 3rd)
        # if (i % 3 == 0):
        plt.subplot(121)
        image_file = image_files[i]
        img = mpimg.imread(image_file)
        plt.imshow(img)

        # plot
        plt.subplot(122)
        plt.plot(range(i,i+len(plotData)), x[:,0], 'r')
        # plt.hold(True)
        # plt.plot(range(i,i+len(plotData)), x[:,1], 'b')
        # plt.plot(range(i,i+len(plotData)), x[:,2], 'g')
        # plt.plot(range(i,i+len(plotData)), x[:,3], 'k')
        plt.plot(range(i,i+len(plotData)), x[:,4], 'y')
        plt.plot(range(i,i+len(plotData)), x[:,5], 'c')
        # plt.plot(range(i,i+len(plotData)), x[:,6], 'm')
        # plt.plot(range(i,i+len(plotData)), x[:,7], 'skyblue')
        # plt.plot(range(i,i+len(plotData)), x[:,8], 'springgreen')
        # plt.plot(range(i,i+len(plotData)), x[:,9], 'orange')
        # plt.plot(range(i,i+len(plotData)), x[:,10], 'maroon')
        # plt.plot(range(i,i+len(plotData)), x[:,11], 'peachpuff')
        # plt.plot(range(i,i+len(plotData)), x[:,12], 'lime')
        # plt.plot(range(i,i+len(plotData)), x[:,13], 'plum')
        # plt.plot(range(i,i+len(plotData)), x[:,14], 'navy')
        # plt.plot(range(i,i+len(plotData)), x[:,15], 'aqua')
        plt.draw()
        # plt.hold(False)

        plt.pause(0.01) # seconds
        i += 1

# prepare training data balanced along axis 
# this ensures the "zero" position does not dominate
def balance(samples):
    paths = [os.path.normpath(i)+"\\data.csv" for i in glob.glob(samples[0])]
    image_files, joystick_values = load_balanced_sample(paths)

    X = np.empty(shape=(image_files.size,Sample.IMG_H,Sample.IMG_W,3),dtype=np.uint8)
    y = []

    for i, filename in enumerate(image_files):
        image = imread(filename)
        vec = resize_image(image)
        X[i] = vec
    for val in joystick_values:
        y.append(val)

    print("Saving to file...")
    X = np.asarray(X)
    y = np.asarray(y)

    print(X.shape)
    print(y.shape)

    np.save("data/x_sbal", X)
    np.save("data/y_sbal", y)

    print("Done!")

# prepare training data
def prepare(samples, augment=True):
    print(f"Preparing data from {samples[0]}")

    y = []

    paths = [os.path.normpath(i) for i in glob.glob(samples[0])]

    numpics = 0

    # for sample in samples: 
    for sample in paths:
        print(sample)

        image_files = load_imgs(sample)
        numpics += len(image_files)

        del sample
        del image_files
        gc.collect()

    print(numpics)

    X = np.empty(shape=(numpics,Sample.IMG_H,Sample.IMG_W,3),dtype=np.uint8)

    idx = 0 # Current image write index - from 0 to numpics

    for sample in paths:
    #for sample in samples:
        print(f"Processing {sample}")

        # load sample
        # image_files, joystick_values = load_sample(os.path.normpath(sample))

        # load condensed sample
        image_files, joystick_values = load_steering_sample(os.path.normpath(sample))

        # add joystick values to y
        print(f"Joystick values shape {joystick_values.shape}")
        y.append(joystick_values)
        

        # load, prepare and add images to X
        for image_file in image_files:
            image = imread(image_file)
            # debug show image
            # plt.imshow(image)
            # plt.show()
            vec = resize_image(image)
            # debug show image
            # plt.imshow(vec)
            # plt.show()
            
            '''
            if augment:
                ## Augmentation
                # Mirror image  
                ### if random.choice([True, False]):
                ###     vec = vec[:, ::-1, :] # horizontally mirror image
                ###     y[-1][0] *= -1 # negate steering value
                # Crop image (by adding black rectangle to mask extraneous details)
                # print(vec.dtype, vec.shape)
                # sys.exit(1)
                vec = cv2.rectangle(img=vec.astype(np.uint8), pt1=(int(0),int(0)), pt2=(int(480), int(90)), color=[0, 0, 0], thickness=cv2.FILLED)
                # Add random jitter to steering values
                ### y[-1][0] += np.random.normal(loc=0, scale=0.01)
                # TODO Add Bias
            '''

            X[idx] = vec

            idx += 1

            del image
            gc.collect()
        # try to do some memory management
        # delete the current sample data since it has been appended to x and y
        del image_files
        del joystick_values
        gc.collect()


    print("Saving to file...")
    X = np.asarray(X)
    y = np.concatenate(y)

    np.save("data/x_f10s", X)
    np.save("data/y_f10s", y)

    print("Done!")

    print(X.shape)
    print(np.asarray(y).shape)
    
    return

def load_data_from_samples(paths, augment=True, debug=False, generator=False):
    INPUT_SHAPE = (Sample.IMG_H, Sample.IMG_W, Sample.IMG_D)

    # for each path, load y data from data.csv
    # 1st column is picture path, 2nd column is steering angle

    # determine number of samples
    num_samples = 0
    for path in paths:
        with open(path + "/data.csv") as f:
            num_samples += sum(1 for _line in f)
    
    # initialize x and y arrays
    if augment:
        num_samples *= 2 # left/right
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
                img = img.astype(np.float32)
                img /= 255.0
                if debug and i % 500 == 0:
                    plt.imshow(img)
                    plt.title(tokens[1]+" "+str(i))
                    plt.show()
                x[i] = img
                y[i] = float(tokens[1])
                if augment:
                    # flip image
                    img = cv2.flip(img, 1)
                    # new index to save at is i + num_samples/2
                    x[i + num_samples//2] = img
                    y[i + num_samples//2] = -float(tokens[1])
                    if debug and i % 500 == 0:
                        plt.imshow(img)
                        new_steering_str = "F"+str(-float(tokens[1]))
                        plt.title(new_steering_str+" "+str(i + num_samples//2))
                        plt.show()
                print(f"sample {i} of {num_samples}", end="\r")
                i += 1
    
                # if generator:
                #     yield x[i], y[i]
    # if not generator:
    #     return x, y
    return x, y


def build_sorted_dataset(dataset_path: str, samples_paths: list):
    # build sorted dataset at samples/path
    # for each sample, put into a folder based on the steering value
    # 15 buckets from -1 to 1
    # closest value goes in that bucket

    # load samples
    x, y = load_data_from_samples(samples_paths)

    # create folders
    buckets = [i/7 for i in range(-7, 8)]
    path_suffixes = [chr(int(i*7 + 72)) for i in buckets]

    # fix dataset_path  
    dataset_path = os.path.normpath(dataset_path)

    for suffix in path_suffixes:
        os.makedirs(os.path.join(dataset_path, suffix))
    
    # sort data
    # hist = plt.hist(y, bins=14)
    print("Sorting data...")
    for i, img in enumerate(x):
        print(i, "/", len(x)-1, end="\r")
        # find closest bucket
        closest = min(buckets, key=lambda x:abs(x-y[i])) # minimize distance between bucket and steering value
        # save image to bucket
        # debug
        # plt.imshow(img)
        # plt.title(f"{i} {y[i]} {closest} {np.max(img)}")
        # plt.show()
        # flip image to BGR for cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dataset_path, chr(int(closest*7 + 72)), str(i)+".jpg"), img*255)

    # plt.show()


if __name__ == '__main__':
    if sys.argv[1] == 'viewer':
        viewer(sys.argv[2])
    elif sys.argv[1] == 'prepare':
        prepare(sys.argv[2:], augment=False)
    elif sys.argv[1] == 'balance':
        balance(sys.argv[2:])
    elif sys.argv[1] == 'plot':
        plot_data(y_pth=sys.argv[2])
    elif sys.argv[1] == 'plotpredictions':
        plot_data(y_pth=sys.argv[2], predictions=True, model_pth=sys.argv[3], x_pth=sys.argv[4], categorical=(sys.argv[5]))
    elif sys.argv[1] == 'seqplotpreds':
        seq_plotpreds(sys.argv[2], sys.argv[3], categorical=True, seq_len=sys.argv[4])
    elif sys.argv[1] == 'show':
        show_pic(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'sort':
        build_sorted_dataset(sys.argv[2], sys.argv[3:])
    else:
        print("(viewer|prepare|balance|plot|plotpredictions|seqplotpreds|show|sort)")