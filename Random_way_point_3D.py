# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:00:50 2024

@author: aydam
"""

import numpy as np
import pandas as pd
import scipy.io

def random_waypoint_3D_for_CHs(nr_nodes, dimensions, initial_positions, speed_low, speed_high, pause_low, pause_high, num_positions_per_node, z_min):
    ndim = 3  # For 3D waypoints
    positions = np.empty((nr_nodes, num_positions_per_node, ndim))
    
    for i in range(nr_nodes):
        current_position = initial_positions[i]
        # Make sure initial z is not 0
        if current_position[2] == 0:
            current_position[2] = z_min
        
        for t in range(num_positions_per_node):
            pause_time = np.random.uniform(pause_low, pause_high)
            speed = np.random.uniform(speed_low, speed_high)
            
            # Generate a random direction in 3D
            azimuth = np.random.uniform(0, 2 * np.pi)
            elevation = np.random.uniform(-np.pi/2, np.pi/2)  # Elevation from -90 to 90 degrees
            
            # Convert spherical to Cartesian coordinates for velocity
            dx = speed * np.cos(elevation) * np.sin(azimuth)
            dy = speed * np.cos(elevation) * np.cos(azimuth)
            dz = speed * np.sin(elevation)
            
            new_position = current_position + np.array([dx, dy, dz])
            
            # Clip the new positions to the dimensions and enforce z >= z_min
            new_position = np.clip(new_position, [0, 0, z_min], dimensions)
            
            positions[i, t] = new_position
            current_position = new_position  # Update current position
            
    return positions


def random_waypoint_2D_for_ground(nr_nodes, dimensions, initial_positions, speed_low, speed_high, pause_low, pause_high, num_positions_per_node):
    ndim = 3  # For 3D positions (x, y, z=0)
    positions = np.empty((nr_nodes, num_positions_per_node, ndim))
    
    for i in range(nr_nodes):
        current_position = initial_positions[i, :2]  # Take only x, y, ignore z if provided
        for t in range(num_positions_per_node):
            pause_time = np.random.uniform(pause_low, pause_high)
            speed = np.random.uniform(speed_low, speed_high)
            
            # Generate a random direction in 2D
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Convert polar to Cartesian coordinates for velocity
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            
            new_position = current_position + np.array([dx, dy])
            
            # Clip the new positions to the dimensions and ensure z=0
            new_position = np.clip(new_position, [0, 0], dimensions[:2])
            new_position = np.append(new_position, 0)  # Append z=0
            
            positions[i, t] = new_position
            current_position = new_position[:2]  # Update current position
            
    return positions

# Example usage:
nr_nodes = 32
num_positions_per_node = 5  # Set the number of positions per node

dimensions = (1000, 1000, 100)  # 3D dimensions (x, y, z)

init_nodes = scipy.io.loadmat('pos_UEs.mat')
initial_positions = init_nodes['intial_positions_UEs']

speed_low = 15.0
speed_high = 30.0
pause_low = 10.0
pause_high = 20.0

# Call the function to generate positions, speeds, and pause times
positions = random_waypoint_2D_for_ground(nr_nodes, dimensions, initial_positions, speed_low, speed_high, pause_low, pause_high, num_positions_per_node)
print(positions)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(positions.reshape(nr_nodes, -1))
df.to_csv("data_mobility_users.csv", header=None, index=None)


# For Random Waypoint
# -------------------

NUMBER_OF_JAMMERS_RW = 8

init_jammers = scipy.io.loadmat('pos_JAMs.mat')
initial_positions_jam = init_jammers['initial_positions_JAMs']

speed_low_jam = 40.0
speed_high_jam = 60.0
pause_low_jam = 10.0
pause_high_jam = 20.0

# Call the function to generate positions, speeds, and pause times
positions_jam = random_waypoint_2D_for_ground(NUMBER_OF_JAMMERS_RW, dimensions, initial_positions_jam, speed_low_jam, speed_high_jam, pause_low_jam, pause_high_jam, num_positions_per_node)
print('positions_jam =', positions_jam)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(positions_jam.reshape(NUMBER_OF_JAMMERS_RW, -1))
df.to_csv("data_mobility_jammers.csv", header=None, index=None)



# For Random Waypoint
# -------------------

dimensions_CH = (1000, 1000, 100)

NUMBER_OF_CHS_RW = 10

init_CHs = scipy.io.loadmat('pos_BS.mat')
initial_positions_CH = init_CHs['initial_positions_CHs']

z_min_CH = 5.0  # Set this to the minimum altitude for CHs

speed_low_CH = 1.0
speed_high_CH = 1.0
pause_low_CH = 4.0
pause_high_CH = 5.0

# Call the function to generate positions, speeds, and pause times
positions_CH = random_waypoint_3D_for_CHs(NUMBER_OF_CHS_RW, dimensions_CH, initial_positions_CH, speed_low_CH, speed_high_CH, pause_low_CH, pause_high_CH, num_positions_per_node, z_min_CH)
print('positions_CHs =', positions_CH)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(positions_CH.reshape(NUMBER_OF_CHS_RW, -1))
df.to_csv("data_mobility_CHs.csv", header=None, index=None)
