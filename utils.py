#!/usr/bin/env python

import sys
import gc
import glob
import array
import os
import random
from PIL import Image
import mss

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
import openvino_test

import cv2
from inputs import get_gamepad
import math
import threading

class Screenshotter(object):
    def __init__(self):
        self.sct = mss.mss()
        self.ie, self.net, self.exec_net, self.output_layer_ir, self.input_layer_ir = openvino_test.start()
        
    def take_screenshot(self):
        # Get raw pixels from the screen
        sct_img = self.sct.grab({  "top":Screenshot.OFFSET_Y,
                                "left": Screenshot.OFFSET_X,
                                "width": Screenshot.SRC_W,
                                "height": Screenshot.SRC_H})
        # Create the Image
        temp = np.array(Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX'))
        #Perform segmentation
        temp = self.convert_to_segmented(temp)

        # DEBUG
        import matplotlib.pyplot as plt

        # Resize
        vec = cv2.resize(temp, (Sample.IMG_W, Sample.IMG_H), interpolation=cv2.INTER_LINEAR_EXACT)
        # Augmentations
        # vec = cv2.rectangle(img=vec.astype(np.uint8), pt1=(int(0),int(0)), pt2=(int(480), int(90)), color=[0, 0, 0], thickness=cv2.FILLED)
        return vec

    def convert_to_segmented(self, img):
        return openvino_test.inference(img, self.ie, self.net, self.exec_net, self.output_layer_ir, self.input_layer_ir, True) 


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
    IMG_W = 480
    IMG_H = 270 
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
            vec = resize_image(image)

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

    np.save("data/x_fh5", X)
    np.save("data/y_fh5", y)

    print("Done!")

    print(X.shape)
    print(np.asarray(y).shape)
    
    return


if __name__ == '__main__':
    if sys.argv[1] == 'viewer':
        viewer(sys.argv[2])
    elif sys.argv[1] == 'prepare':
        prepare(sys.argv[2:], augment=False)
    elif sys.argv[1] == 'prepare_augment':
        prepare(sys.argv[2:])
