#!/usr/bin/env python

import time
from PIL import Image
import mss
from mss import screenshot
import tensorflow as tf
from utils import Screenshot, resize_image, XboxController
# from termcolor import cprint

# import gym
# import gym_mupen64plus
# from train import create_model
import openvino_test

try:
    import numpy as np
except ImportError:
    print("Cupy not installed.")
    import numpy as np


import pyxinput
import ctypes

class Screenshotter(object):
    def __init__(self):
        self.sct = mss.mss()
        
    def take_screenshot(self):
        # Get raw pixels from the screen
        sct_img = self.sct.grab({  "top": Screenshot.OFFSET_Y,
                                "left": Screenshot.OFFSET_X,
                                "width": Screenshot.SRC_W,
                                "height": Screenshot.SRC_H})
        # Create the Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')


# Play
class Actor(object):

    def __init__(self):
        # Load in model from train.py and load in the trained weights
        # self.model = create_model(keep_prob=1) # no dropout
        # self.model.load_weights('model_weights_f1_7.h5') # CHANGE THIS WITH A NEW MODEL 

        # Init contoller for manual override
        self.real_controller = XboxController()

        # Init fake controller
        self.controller = pyxinput.vController()

        self.thresh = 0.04 # threshold for magnifying value

        self.cutoff = 0.4 # cutoff value for the steering (CV)

        self.anglefactor = 1/3 # coeff for steering (CV)

        self.ie, self.net, self.exec_net, self.output_layer_ir, self.input_layer_ir = openvino_test.start()

        self.lastvalue = 0 # in case there is an error

    # HIGH LEVEL CONTROL METHODS
    def shoot(self): # NOTE 0 
        self.controller.set_value("BtnX",1)
        time.sleep(0.56) # https://www.nba2klab.com/2k21/meter
        self.controller.set_value("BtnX",0)

    def pass_ball(self): # NOTE 1
        self.controller.set_value("BtnA",1)
        # don't need to reset to 0 because of loop?

    def move_u(self): # NOTE Don't know?
        self.controller.set_value("TriggerR",1)
        self.controller.set_value("AxisLy",1)

    def move_d(self): # NOTE Don't know?
        self.controller.set_value("TriggerR",1)
        self.controller.set_value("AxisLy",-1)

    def move_r(self): # NOTE 2
        self.controller.set_value("TriggerR",1)
        self.controller.set_value("AxisLx",1)

    def move_l(self): # NOTE 3
        self.controller.set_value("TriggerR",1)
        self.controller.set_value("AxisLx",-1)

    def block(self):
        self.controller.set_value("TriggerL",1)
        self.controller.set_value("TriggerR",1)
        self.controller.set_value("BtnY",1)

    def reset_controller(self):
        self.controller.set_value("AxisLx",0)
        self.controller.set_value("AxisLy",0)
        self.controller.set_value("AxisRx",0)
        self.controller.set_value("AxisRy",0)
        self.controller.set_value("TriggerL",0)
        self.controller.set_value("TriggerR",0)
        self.controller.set_value("BtnShoulderL",0)
        self.controller.set_value("BtnShoulderR",0)
        self.controller.set_value("BtnA",0)
        self.controller.set_value("BtnX",0)
        self.controller.set_value("BtnY",0)
        self.controller.set_value("BtnB",0)
        self.controller.set_value("BtnThumbL",0)
        self.controller.set_value("BtnThumbR",0)
        self.controller.set_value("BtnBack",0)
        self.controller.set_value("BtnStart",0)

    def control(self, joystick):
        self.controller.set_value("AxisLx",joystick[0])
        self.controller.set_value("AxisLy",joystick[1])
        self.controller.set_value("AxisRx",joystick[2])
        self.controller.set_value("AxisRy",joystick[3])
        self.controller.set_value("TriggerL",joystick[4])
        self.controller.set_value("TriggerR",joystick[5])
        self.controller.set_value("BtnShoulderL",joystick[6])
        self.controller.set_value("BtnShoulderR",joystick[7])
        self.controller.set_value("BtnA",joystick[8])
        self.controller.set_value("BtnX",joystick[9])
        self.controller.set_value("BtnY",joystick[10])
        self.controller.set_value("BtnB",joystick[11])
        self.controller.set_value("BtnThumbL",joystick[12])
        self.controller.set_value("BtnThumbR",joystick[13])
        # self.controller.set_value("BtnBack",joystick[14])
        self.controller.set_value("BtnBack",0)
        # self.controller.set_value("BtnStart",joystick[15])
        self.controller.set_value("BtnStart",1)

    def control_mini(self, joystick):
        self.controller.set_value("AxisLx",joystick[0])
        self.controller.set_value("AxisLy",joystick[1])
        self.controller.set_value("AxisRx",joystick[2])
        self.controller.set_value("AxisRy",joystick[3])
        self.controller.set_value("TriggerL",joystick[4])
        self.controller.set_value("TriggerR",joystick[5])
        self.controller.set_value("BtnA",joystick[6])
        self.controller.set_value("BtnX",joystick[7])

    def control_racing(self, joystick):
        self.controller.set_value("AxisLx",joystick[0])
        # self.controller.set_value("TriggerL",joystick[1])
        self.controller.set_value("TriggerR",joystick[1])

    def cv_act_racing(self, img): 
        manual_override = self.real_controller.RightThumb == 1
        if not manual_override:
            angle = openvino_test.inference(np.array(img), self.ie, self.net, self.exec_net, self.output_layer_ir, self.input_layer_ir)
            if angle == "error":
                joystick = [-self.lastvalue, 0.3]
                print(f"ERROR: Last angle is used: {-self.lastvalue}")
                self.control_racing(joystick)
                return
            
            angle *= self.anglefactor
            self.lastvalue = angle
            
            print(angle)
            if angle > self.cutoff:
                angle = self.cutoff
            elif angle < -self.cutoff:
                angle = -self.cutoff
            joystick = [angle, 0.3]

            self.control_racing(joystick)
        
    def act(self, img):

        ## determine manual override
        manual_override = self.real_controller.RightThumb == 1

        if not manual_override:
        # if True:
            ## Look
            print("debug")
            # print(img)
            vec = resize_image(np.array(img))
            vec = np.expand_dims(vec, axis=0) # expand dimensions for predict, it wants (1,66,200,3) not (66, 200, 3)
            ## Think
            joystick = self.model.predict(vec, batch_size=1)[0]
            
            # for i,num in enumerate(joystick):
            #     if joystick[i] >= self.thresh: # predicting 1, positive
            #         joystick[i] = 1
            #     elif -self.thresh < joystick[i] < self.thresh: # predicting 0, nothing
            #         joystick[i] = 0
            #     elif joystick[i] <= -self.thresh: # predicting -1, negative
            #         joystick[i] = -1
            #     else:
            #         print("Error")

            joystick = joystick * 3
            for i,num in enumerate(joystick):
                if joystick[i] > 1:
                    joystick[i] = 1
                elif joystick[i] < -1:
                    joystick[i] = -1
                else:
                    pass
            
            print(joystick)

            # if len(joystick) == 8:
            #     self.control_mini(joystick)
            # elif len(joystick) == 16:
            #     self.control(joystick)
            # else:
            #     print("Invalid Joystick Length")
            self.control_racing(joystick)
        # 
        else:
            print("Manual Override")
            joystick = self.real_controller.read()
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)

            self.control(joystick)

        
        ## Act
        ## has been put in Think block NOTE 


if __name__ == '__main__':
    # set this program to higher priority (for realtime shooting etc)
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 14) 

    #disable gpu
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    
    # turn on program and let it run forever
    actor = Actor()
    screenshot = Screenshotter()
    print('actor ready!')
    while True:
        pic = screenshot.take_screenshot()
        actor.cv_act_racing(pic)

