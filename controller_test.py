# for testing the controller lib
# Enter a game and run this program. 
# The virtual controller should start entering inputs into the game. 
# ControlC to exit the program. 

import pyxinput
import time
controller = pyxinput.vController()

while True:
    controller.set_value("AxisLy",1)
    controller.set_value("AxisLx",1)
    controller.set_value("AxisRy",1)
    controller.set_value("AxisRx",1)
    controller.set_value("TriggerL",1)
    controller.set_value("TriggerR",1)
    controller.set_value("BtnA",1)
    controller.set_value("BtnX",1)
    controller.set_value("BtnStart",1)
    controller.set_value("BtnB",1)
    time.sleep(0.5)
    controller.set_value("AxisLy",0)
    controller.set_value("AxisLx",0)
    controller.set_value("AxisRy",0)
    controller.set_value("AxisRx",0)
    controller.set_value("TriggerL",0)
    controller.set_value("TriggerR",0)
    controller.set_value("BtnA",0)
    controller.set_value("BtnX",0)
    controller.set_value("BtnStart",0)
    controller.set_value("BtnB",0)