# for testing the controller lib
"""
import gamePyd

gamepad = gamePyd.wPad()

while True:
    for i in range(-32768,32768):
        gamepad.set_value("AxisLx",i)

"""
"""
import pyvjoy

j = pyvjoy.VJoyDevice(1) 

while True:
    j.data.wAxisX = 0x2000 
    j.data.wAxisY= 0x7500
    j.update()
    j.reset() 
    """

import pyxinput
import time
controller = pyxinput.vController()

while True:
    # controller.set_value("AxisLy",1)
    controller.set_value("AxisLx",1)
    # controller.set_value("AxisRy",1)
    # controller.set_value("AxisRx",1)
    # controller.set_value("TriggerL",1)
    controller.set_value("TriggerR",1)
    # controller.set_value("BtnA",1)
    # controller.set_value("BtnX",1)
    # controller.set_value("BtnStart",1)
    # controller.set_value("BtnB",1)
    time.sleep(0.5)
    # controller.set_value("AxisLy",0)
    controller.set_value("AxisLx",0)
    # controller.set_value("AxisRy",0)
    # controller.set_value("AxisRx",0)
    # controller.set_value("TriggerL",0)
    controller.set_value("TriggerR",0)
    # controller.set_value("BtnA",0)
    # controller.set_value("BtnX",0)
    # controller.set_value("BtnStart",0)
    # controller.set_value("BtnB",0)

# import time
# import random
# from play import Actor
# 
# actor = Actor()
# actor.controller.set_value("BtnB",1)
# time.sleep(1)
# actor.controller.set_value("BtnStart",1)
# print("started")
# 
# 
# for i in range(200):
#     actor.controller.set_value("BtnX",1)
#     time.sleep(1)
#     actor.controller.set_value("BtnStart",bool(round(random.random())))
#     actor.controller.set_value("BtnB",1)
#     actor.shoot()
#     print("shot ball")
#     actor.reset_controller()
#     print("reset")
#     actor.controller.set_value("BtnY",1)
#     actor.controller.set_value("BtnA",1)
#     time.sleep(0.4)
#     actor.controller.set_value("AxisRy",1)
#     time.sleep(1)