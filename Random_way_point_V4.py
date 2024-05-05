import numpy as np
import pandas as pd
import scipy.io

def random_waypoint_with_initial_positions(nr_nodes, dimensions, initial_positions, speed_low, speed_high, pause_low, pause_high, num_positions_per_node):
    ndim = len(dimensions)
    border_margin = 0.001
    positions = np.empty((nr_nodes, num_positions_per_node, ndim))
    speeds = np.empty((nr_nodes, num_positions_per_node))
    pause_times = np.empty((nr_nodes, num_positions_per_node))
    
    for i in range(nr_nodes):
        current_position = initial_positions[i]
        for t in range(num_positions_per_node):
            pause_time = np.random.uniform(pause_low, pause_high)
            speed = np.random.uniform(speed_low, speed_high)
            
            direction = np.random.uniform(0, 2 * np.pi)  # Random direction in radians
            dx = speed * np.sin(direction)
            dy = speed * np.cos(direction)
            
            new_position = current_position + np.array([dx, dy])
            
            # Ensure the new position stays within the dimensions
            for k in range(ndim):
                new_position[k] = max(0, min(new_position[k], dimensions[k]))
                
                # Modify direction to move away from the border if necessary
                if new_position[k] < dimensions[k] * border_margin:
                    direction = np.random.uniform(0.5 * np.pi, 1.5 * np.pi)
                    dx = speed * np.sin(direction)
                    dy = speed * np.cos(direction)
                    new_position = current_position + np.array([dx, dy])
                elif new_position[k] > dimensions[k] * (1 - border_margin):
                    direction = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
                    dx = speed * np.sin(direction)
                    dy = speed * np.cos(direction)
                    new_position = current_position + np.array([dx, dy])
            
            # Ensure that the new position is non-negative
            new_position = np.maximum(0, new_position)
            
            positions[i, t] = new_position
            speeds[i, t] = speed
            pause_times[i, t] = pause_time
            current_position = new_position
            
    return positions, speeds, pause_times

# Example usage:
nr_nodes = 38
num_positions_per_node = 5  # Set the number of positions per node

dimensions = (1000, 1000)  # Adjust the dimensions as needed

init_nodes = scipy.io.loadmat('pos_UEs_10.mat')
initial_positions = init_nodes['intial_positions_UEs']

speed_low = 15.0
speed_high = 30.0
pause_low = 10.0
pause_high = 20.0

# Call the function to generate positions, speeds, and pause times
positions, speeds, pause_times = random_waypoint_with_initial_positions(nr_nodes, dimensions, initial_positions, speed_low, speed_high, pause_low, pause_high, num_positions_per_node)
print(positions)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(positions.reshape(nr_nodes, -1))
df.to_csv("data_mobility_users_CH10.csv", header=None, index=None)



# For Random Waypoint
# -------------------

NUMBER_OF_JAMMERS_RW = 5

init_jammers = scipy.io.loadmat('pos_JAMs_10.mat')
initial_positions_jam = init_jammers['initial_positions_JAMs']

speed_low_jam = 40.0
speed_high_jam = 60.0
pause_low_jam = 10.0
pause_high_jam = 20.0

# Call the function to generate positions, speeds, and pause times
positions_jam, speeds, pause_times = random_waypoint_with_initial_positions(NUMBER_OF_JAMMERS_RW, dimensions, initial_positions_jam, speed_low_jam, speed_high_jam, pause_low_jam, pause_high_jam, num_positions_per_node)
print('positions_jam =', positions_jam)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(positions_jam.reshape(NUMBER_OF_JAMMERS_RW, -1))
df.to_csv("data_mobility_jammers_CH10.csv", header=None, index=None)



# For Random Waypoint
# -------------------

NUMBER_OF_CHS_RW = 10

init_CHs = scipy.io.loadmat('pos_BS_10.mat')
initial_positions_CH = init_CHs['initial_positions_CHs']

speed_low_CH = 1.0
speed_high_CH = 1.0
pause_low_CH = 5.0
pause_high_CH = 6.0

# Call the function to generate positions, speeds, and pause times
positions_CH, speeds, pause_times = random_waypoint_with_initial_positions(NUMBER_OF_CHS_RW, dimensions, initial_positions_CH, speed_low_CH, speed_high_CH, pause_low_CH, pause_high_CH, num_positions_per_node)
print('positions_CHs =', positions_CH)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(positions_CH.reshape(NUMBER_OF_CHS_RW, -1))
df.to_csv("data_mobility_CHs_CH10.csv", header=None, index=None)
