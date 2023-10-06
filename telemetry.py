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
    # save_waypoints()
    # pid_throttle()

    #for i in get_waypoints(print_values=False):
    #    x, y, z, speed = i
    #    print(f"x: {x}, y: {y}, z: {z}, speed: {speed}")
    #    # time.sleep(1) 

    follow_waypoints()
        