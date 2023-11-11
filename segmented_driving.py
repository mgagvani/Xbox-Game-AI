import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# Load the CSV data
data = pd.read_csv("forza_motorsport/forza_data2.csv")

# Define the state space discretization parameters
x_bins = np.linspace(min(data['position_x']), max(data['position_x']), num=50)
y_bins = np.linspace(min(data['position_y']), max(data['position_y']), num=50)
speed_bins = np.linspace(min(data['speed']), max(data['speed']), num=5)

# Create a grid for all possible state combinations
grid_x, grid_y, grid_speed = np.meshgrid(x_bins, y_bins, speed_bins)

# Define a function to find the nearest grid point
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array.flat[idx]

# Initialize lookup tables for control inputs (steering and throttle)
steer_lookup = np.zeros_like(grid_x)
throttle_lookup = np.zeros_like(grid_x)

# Populate the lookup tables with interpolated values
for idx, row in data.iterrows():
    x = find_nearest(x_bins, row['position_x'])
    y = find_nearest(y_bins, row['position_y'])
    speed = find_nearest(speed_bins, row['speed'])
    steer = row['steer']
    throttle = row['accel']
    
    # Find the indices of the nearest grid point
    x_idx = np.where(x_bins == x)[0][0]
    y_idx = np.where(y_bins == y)[0][0]
    speed_idx = np.where(speed_bins == speed)[0][0]
    
    # Populate the lookup tables
    steer_lookup[y_idx, x_idx, speed_idx] = steer
    throttle_lookup[y_idx, x_idx, speed_idx] = throttle

# Function to estimate control inputs based on current state
def estimate_controls(x, y, speed):
    x_idx = np.where(x_bins == find_nearest(x_bins, x))[0][0]
    y_idx = np.where(y_bins == find_nearest(y_bins, y))[0][0]
    speed_idx = np.where(speed_bins == find_nearest(speed_bins, speed))[0][0]
    
    steer = steer_lookup[y_idx, x_idx, speed_idx]
    throttle = throttle_lookup[y_idx, x_idx, speed_idx]
    
    return steer, throttle

def visualize_lookup_table(lookup_table, x_bins, y_bins, speed_bins):
    # Create a meshgrid for the state space
    grid_x, grid_y, grid_speed = np.meshgrid(x_bins, y_bins, speed_bins, indexing='ij')

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(grid_x[:, :, 0], grid_y[:, :, 0], lookup_table[:, :, 0], cmap='jet')
    plt.colorbar(label='Steering Angle')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Steering Angle Lookup Table')
    plt.gca().invert_yaxis()  # Invert Y axis to match typical coordinate systems
    plt.show()

def plot_variables(variables: list):
    plt.figure(figsize=(12, 6))
    for variable in variables:
        plt.plot(data[variable], label=variable)
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('Variable')
    plt.legend()
    plt.show()

def plot_speed_2d():
    # create 2d figure with speed colormapped
    plt.figure(figsize=(12, 6))
    plt.scatter(data['position_x'], data['position_y'], c=data['speed'], cmap='jet', s=1)
    plt.colorbar(label='Speed (m/s)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Speed')
    plt.gca().invert_yaxis()  # Invert Y axis to match typical coordinate systems
    plt.show()

def live_plot():
    import sys, matplotlib
    sys.path.append('forza_motorsport/')
    from data2file import return_vals
    # matplotlib.use('GTK4Agg')

    rw = return_vals(2560)

    # continuously growing line polot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value')
    ax.set_title('Variable')
    line1 = ax.plot([], [], 'r-')[0]
    # line2 = ax.plot([], [], 'b-')[0] # steer
    line3 = ax.plot([], [], 'g-')[0]
    line4 = ax.plot([], [], 'k-')[0]
    lines = [line1, line3, line4]

    # legend
    ax.legend(lines, ["norm_driving_line", "accel", "brake"])

    x = 0
    while True:
        x += 1
        for i, line in enumerate(lines):
            line.set_xdata(np.append(line.get_xdata(), x))
            line.set_ydata(np.append(line.get_ydata(), next(rw)[i]))
        
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    

if __name__ == "__main__":
    live_plot()

    plot_variables(["steer", "accel", "brake", "speed", "norm_driving_line"])

    plot_speed_2d()

    # Example usage to estimate control inputs
    x, y, speed = int(input("x: ")), int(input("y: ")), int(input("speed: "))
    steer, throttle = estimate_controls(x, y, speed)
    print(f"Steering: {steer}, Throttle: {throttle}, x: {x}, y: {y}, speed: {speed}")

    # Example usage to visualize the steering lookup table
    visualize_lookup_table(steer_lookup, x_bins, y_bins, speed_bins)
