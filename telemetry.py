import sys
sys.path.append("forza_motorsport/")

import numpy as np
from simple_pid import PID

from data2file import return_vals
from play import Actor

PORT_NUMBER = 2560 # change to the port number specified in Forza

def save_waypoints():
    generator = return_vals(PORT_NUMBER)
    vals = []

    try:
        for data in generator:
            vals.append(data)
    except KeyboardInterrupt:
        print("CtrlC detected, stopping ands saving")
        np.savetxt("waypoints.txt", vals)
    
def get_waypoints(print=False):
    generator = return_vals(PORT_NUMBER)
    
    for i in generator:
        if print:
            print(i)
        else:
            yield i

class pid_throttle():
    def __init__(self):
        self.SPEED = 20 # m/s
        self.pid = PID(1, 1, 1, setpoint=self.SPEED)
        # self.pid.sample_time = 1/5 # assume 5 FPS
        self.pid.output_limits = (0, 1)
        self.speed_generator = return_vals(PORT_NUMBER)

    def get_speed(self):
        speed = next(self.speed_generator)[-1]
        out = self.pid(speed)
        return out

if __name__ == "__main__":
    actor = Actor(load_model=False)
    pidt = pid_throttle()

    while True:
        try:
            actor.control_throttle(pidt.get_speed())
        except KeyboardInterrupt:
            print("Stopping due to Ctrl C Event")
            break