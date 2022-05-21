import sys
sys.path.append("forza_motorsport/")

import numpy as np

from data2file import return_vals

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
    
def print_waypoints():
    generator = return_vals(PORT_NUMBER)
    
    for i in generator:
        print(i)

if __name__ == "__main__":
    save_waypoints()