#!/usr/bin/env python

# Author: Mihir Singh
# Date: 2025/05/20

# imports
import numpy as np
import tf_transformations
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseArray, Twist
from collections import deque, defaultdict
from typing import List, Tuple, Dict
import tf2_ros
import math
from rclpy.duration import Duration
import csv

# topics
NODE_NAME = "ValueIteration"
MAP_TOPIC = "map"

# ref frames
MAP_FRAME_ID = "map"

# robot constants
USE_SIM_TIME = True
FREQUENCY = 10 #Hz.
ANGULAR_VELOCITY = np.pi/4 # rad/s
LINEAR_VELOCITY = 0.125 # m/s

# value iteration constants
# actions
ACTIONS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'Stay': (0, 0)
}

# probabilities for actions 
DIRECTION_PROBABILITIES = {
    'N': [('N', 1.0), ('E', 0.0), ('W', 0.0)],
    'S': [('S', 1.0), ('E', 0.0), ('W', 0.0)],
    'E': [('E', 1.0), ('N', 0.0), ('S', 0.0)],
    'W': [('W', 1.0), ('N', 0.0), ('S', 0.0)],
    'Stay': [('Stay', 1.0)]
}

class Grid:
    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.resolution = resolution
        self.width = width
        self.height = height

    # returns value of cell at (r, c)
    def cell_at(self, r, c):
        return self.grid[r, c]
    
    # checks if cell is a valid cell for the robot to move to
    def is_valid(self, r, c):
        # check if coordinate itself is out of bounds
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return False

        # check if cell is wall
        if self.cell_at(r, c) == 100:
            return False

        # check all cells within the buffer distance (in pixels)
        buffer_radius = 7
        
        # find search boundaries with buffer
        min_row = max(0, r - buffer_radius)
        max_row = min(self.height - 1, r + buffer_radius)
        min_col = max(0, c - buffer_radius)
        max_col = min(self.width - 1, c + buffer_radius)
        
        # check all cells within the buffer radius 
        for rr in range(min_row, max_row + 1):
            for cc in range(min_col, max_col + 1):
                # euclidean distance
                distance = np.sqrt((rr - r)**2 + (cc - c)**2)
                if distance <= buffer_radius and self.cell_at(rr, cc) == 100:
                    return False

        return True

class ValueIteration(Node):
    def __init__(self, map_frame_id=MAP_FRAME_ID, node_name=NODE_NAME, context=None):
        super().__init__(node_name, context=context)
        
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])
        
        # Rate at which to operate the while loop.
        self.rate = self.create_rate(FREQUENCY)
        
        # for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # subscribers
        self._map_sub = self.create_subscription(OccupancyGrid, MAP_TOPIC, self.map_callback, 10)
        
        # robot constants
        self.linear_velocity = LINEAR_VELOCITY
        self.angular_velocity = ANGULAR_VELOCITY
        
        # init constants
        self.map = None
        self.map_frame_id = map_frame_id
        self.map_origin = None
        self.occupancy_grid = None
        
    # get and update map
    def map_callback(self, msg):
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        self.map_origin = msg.info.origin
        self.occupancy_grid = msg
        print(f"Got map: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}")
    
    # world to grid conversion 
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        x_rel = x - self.map_origin.position.x
        y_rel = y - self.map_origin.position.y
        col = int(x_rel / self.map.resolution)
        row = int(y_rel / self.map.resolution)
        return row, col

    # grid to world conversion 
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = col * self.map.resolution + self.map_origin.position.x + self.map.resolution / 2
        y = row * self.map.resolution + self.map_origin.position.y + self.map.resolution / 2
        return x, y
    
    # get block-based states (each state is a 5x5 block)
    def get_states(self):
        states = []
        block_size = 20
        for block_r in range((self.map.height + block_size - 1) // block_size):
            for block_c in range((self.map.width + block_size - 1) // block_size):
                states.append((block_r, block_c))
        return states

    # define rewards for block-based states
    def get_rewards(self, goal: Tuple[int, int]):
        rewards = {}
        block_size = 5
        goal_block = (goal[0] // block_size, goal[1] // block_size)
        for block_r in range((self.map.height + block_size - 1) // block_size):
            for block_c in range((self.map.width + block_size - 1) // block_size):
                # Check if any cell in the 5x5 block is invalid
                is_block_valid = True
                for r in range(block_r * block_size, min((block_r + 1) * block_size, self.map.height)):
                    for c in range(block_c * block_size, min((block_c + 1) * block_size, self.map.width)):
                        if not self.map.is_valid(r, c):
                            is_block_valid = False
                            break
                    if not is_block_valid:
                        break
                # Assign rewards
                if (block_r, block_c) == goal_block and is_block_valid:
                    rewards[(block_r, block_c)] = 10  # Goal reward (only if block is valid)
                elif not is_block_valid:
                    rewards[(block_r, block_c)] = -100  # Penalty for blocks with walls
                else:
                    rewards[(block_r, block_c)] = -0.1  # Step cost
        return rewards
    
    # helps keep track of boundaries (unused)
    def next_state(self, state_idx, direction):
        pos = np.argwhere(self.occupancy_grid == state_idx)[0]
        delta = ACTIONS[direction]
        new_pos = pos + delta
        if 0 <= new_pos[0] < self.occupancy_grid.shape[0] and 0 <= new_pos[1] < self.occupancy_grid.shape[1]:
            return self.occupancy_grid[tuple(new_pos)]
        return state_idx  # stay in place if moving off the grid

    def value_iteration(self, start: Tuple[int, int], goal: Tuple[int, int], discount_factor: float = 0.9, theta: float = 0.01):
        # Get states and rewards
        block_size = 5
        start_block = (start[0] // block_size, start[1] // block_size)
        goal_block = (goal[0] // block_size, goal[1] // block_size)
        states = self.get_states()
        rewards = self.get_rewards(goal)

        # Validate state space
        if not states:
            self.get_logger().error("No valid states found in the grid!")
            return {}, {}

        # Initialize value function and policy
        V = {state: 0.0 for state in states}
        policy = {state: None for state in states}

        # Add iteration limit to prevent infinite loops
        max_iterations = 1000
        iteration = 0

        # Value iteration
        while True:
            delta = 0.0  # Tracks max change in value function
            for state in states:
                if state == goal_block:  # Skip terminal state
                    V[state] = rewards.get(state, 10)  # Ensure goal has fixed reward
                    continue
                old_v = V[state]
                action_values = {}
                for action, move in ACTIONS.items():
                    value = 0
                    for direction, prob in DIRECTION_PROBABILITIES[action]:
                        # Compute next block state
                        next_block_r, next_block_c = state[0] + move[0], state[1] + move[1]
                        next_state = (next_block_r, next_block_c)
                        # Check if next state is valid; if not, stay in place
                        if (0 <= next_block_r < (self.map.height + block_size - 1) // block_size and 
                            0 <= next_block_c < (self.map.width + block_size - 1) // block_size):
                            value += prob * (rewards.get(next_state, -100) + discount_factor * V.get(next_state, 0))
                        else:
                            value += prob * (rewards.get(state, -100) + discount_factor * V.get(state, 0))
                    action_values[action] = value
                # Update value and policy
                V[state] = max(action_values.values())
                policy[state] = max(action_values, key=action_values.get)
                delta = max(delta, abs(old_v - V[state]))
            
            iteration += 1
            self.get_logger().info(f"Iteration {iteration}: delta = {delta}")
            
            # Check convergence or max iterations
            if delta < theta:
                self.get_logger().info(f"Converged after {iteration} iterations with delta = {delta}")
                break
            if iteration >= max_iterations:
                self.get_logger().warning(f"Stopped after {max_iterations} iterations without convergence (delta = {delta})")
                break

        return V, policy
    
    def spin(self):
        # Example start and goal positions
        start = (2, 2)
        goal = (1, 1)
        print("Starting Value Iteration...")
        V, policy = self.value_iteration(start, goal)
        # Print the value function and policy
        print("Value Function:")
        for state, value in V.items():
            print(f"Block {state}: {value}")
        print("\nPolicy:")
        for state, action in policy.items():
            print(f"Block {state}: {action}")
            
        # Save Policy to CSV
        with open('policy.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State', 'Action'])  # Write header
            for state, action in policy.items():
                writer.writerow([str(state), action])  # Convert tuple to string
        
        

def main(args=None):
    rclpy.init(args=args)
    vi = ValueIteration()
    
    # wait to see if the map is received
    while vi.occupancy_grid is None:
        rclpy.spin_once(vi)
    if vi.occupancy_grid is not None:
        rclpy.spin_once(vi)
        vi.spin()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
