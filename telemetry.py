import sys
from typing import Any
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

def get_waypoints(print_values=False):
    generator = return_vals(PORT_NUMBER)

    for i in generator:
        if print_values:
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

def pid_throttle():
    actor = Actor(load_model=False)
    pidt = pid_throttle()

    while True:
        try:
            actor.control_throttle(pidt.get_speed())
        except KeyboardInterrupt:
            print("Stopping due to Ctrl C Event")
            break

class pid_steer_throttle():
    MPH_TO_MPS = 0.44704

    def __init__(self):
        self.SPEED = 75 * self.MPH_TO_MPS # m/s
        self.throttle_pid = PID(1, 1, 1, setpoint=self.SPEED)
        self.throttle_pid.output_limits = (0, 1)
        self.speed = 0

        self.SETPOINT = 0 # center of the road
        self.S_KP = 4.0 / 127.0
        self.S_KI = 0.5 / 127.0
        self.S_KD = 8.0 / 127.0
        self.steer_pid = PID(self.S_KP, self.S_KI, self.S_KD, setpoint=self.SETPOINT)
        self.steer_pid.output_limits = (-1, 1)
        self.norm_driving_line = 0

        self.generator = return_vals(PORT_NUMBER) # norm_driving_line, speed

        self.v_x, self.v_y, self.v_z = 0, 0, 0 # linear velocity
        self.w_x, self.w_y, self.w_z = 0, 0, 0 # angular velocity

        # if radius of curvature is less than, we are turning hard (close to turning circle)
        # so cut the throttle
        self.RADIUS_THRESH = 35 # m

        self.actor = Actor(load_model=False)
        
    def __update(self):
        data = next(self.generator)
        self.speed = data[1]
        self.norm_driving_line = data[0]
        self.v_x, self.v_y, self.v_z = data[5:8]
        self.w_x, self.w_y, self.w_z = data[2:5]
        assert type(self.speed) is float and type(self.norm_driving_line) in (float, int), f"Error (Assert): {self.speed}, {self.norm_driving_line}"

        # instantaneous radius of curvature
        v = np.linalg.norm([self.v_x, self.v_y, self.v_z])
        w = np.linalg.norm([self.w_x, self.w_y, self.w_z])
        radius = v / w if w != 0 else 0
        
        steer =  - self.steer_pid(self.norm_driving_line)
        throttle = self.throttle_pid(self.speed) if radius > self.RADIUS_THRESH else 0

        # debug - print current steer, norm_driving_line
        # print(f"Steer: {steer}, Norm Driving Line: {self.norm_driving_line}, data: {data}")
        print(f"Velocity: {v}, AngVelocity: {w}, Radius: {radius}")
    
        return steer, throttle
    
    def __call__(self) -> Any:
        while True:
            try:
                steer, throttle = self.__update()
                self.actor.control_racing([steer, throttle])
            except KeyboardInterrupt:
                print("Stopping due to Ctrl C Event")
                break
            # except Exception as e:
            #     print(f"Error (Loop): {e}")
    
def follow_waypoints(waypoints_file="waypoints.txt"):
    actor = Actor(load_model=False)

    sys.path.append("../PythonRobotics/PathTracking/pure_pursuit/")
    import pure_pursuit as pp

    waypoints = np.loadtxt(waypoints_file) # x, y, z, speed
    cx = waypoints[:, 0]
    cy = waypoints[:, 1]
    cv = waypoints[:, 3]
    target_course = pp.TargetCourse(cx, cy)
    state = pp.State(x=-0.0, y=-3.0, yaw=np.deg2rad(0.0), v=0.0)
    trajectory = pp.Trajectory(cx, cy, 0.1)
    target_ind, _ = target_course.search_target_index(state)

    while True:
        try:
            target_ind, _ = target_course.search_target_index(state)
            di, target_ind = pp.pure_pursuit_steer_control(state, trajectory, target_ind)
            ai = pp.proportional_control(cv[target_ind], state.v)
            state.update(ai, di)
            # di is the steering angle.
            # update with controller
            throttle = 1 if state.v < 20 else 0
            # clip 
            di = np.clip(di, -1, 1)
            actor.control_racing([di, throttle])

        except KeyboardInterrupt:
            print("Stopping due to Ctrl C Event")
            break



if __name__ == "__main__":
    pid_steer_throttle()()
