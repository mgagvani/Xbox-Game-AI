"""Simple example showing how to get gamepad events."""

from __future__ import print_function


from inputs import get_gamepad

from utils import XboxController
import time
import matplotlib.pyplot as plt 
import matplotlib


def main():
    """Just print out some event infomation when the gamepad is used."""
    while 1:
        events = get_gamepad()
        for event in events:
            # print(event.ev_type, event.code, event.state)
            print(event.code)

def main2():
    # test XboxController
    controller = XboxController()
    print('init')

    values = {}

    # determine average polling rate
    start = time.time()
    count = 0
    while True:
        try:
            t = time.time() - start
            values[t] = controller.read()
            print(values[t])
            count += 1
        except KeyboardInterrupt:
            print("Stopping due to Ctrl C Event")
            break
    end = time.time()
    print(f"Average polling rate: {count/(end-start)} Hz ({count} events in {end-start} seconds)")

    # plot everything on one graph
    plt.figure()
    plt.title("Xbox Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    pairs = [(k, v) for k, v in values.items()]
    plt.plot([i[0] for i in pairs], [i[1] for i in pairs])
    plt.savefig("xboxcontroller.png")



if __name__ == "__main__":
    matplotlib.use('Agg')
    main2()
