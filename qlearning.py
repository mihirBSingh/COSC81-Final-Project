import time
import numpy as np
import random
import rclpy
from mapping import GridMapper
from rclpy.node import Node
import threading
import os
from geometry_msgs.msg import PoseArray, Pose
import tf_transformations
import math

actions_list = ["UP", "RIGHT", "DOWN", "LEFT"]

def print_grid_window(map_data, center, window_size=5):
    cx, cy = center
    half = window_size // 2

    print(f"\n--- Grid window around {center} ---")
    for y in range(cy - half, cy + half + 1):
        row = ""
        for x in range(cx - half, cx + half + 1):
            if 0 <= x < map_data.shape[0] and 0 <= y < map_data.shape[1]:
                val = map_data[y, x]
                if (x, y) == center:
                    row += f"[{val:3}] "  # Highlight center
                else:
                    row += f" {val:3}  "
            else:
                row += "  X   "
        print(row)
    print("--- End Grid ---\n")

def print_qtable_window(q_table, state):
    """
    Print the Q-values for all actions at a single state.
    Args:
        q_table: 3D numpy array (x, y, actions)
        state: tuple of (x, y) coordinates
    """
    
    print(f"\n=== Q-values at state {state} ===")
    for action, name in enumerate(actions_list):
        x, y = state
        print(f"{name:5}: {q_table[y][x][action]:6.2f}")
    
    print("=== End Q-values ===\n")


class QLearningAgent(Node):
    def __init__(self, discount_rate=0.9, learning_rate=0.1, exploration_rate=0.3, initial_size=1000, res=0.05):
        super().__init__('qlearning_agent')  # Initialize the ROS node
        grid_size = int(initial_size * res)
        self.q_table = np.zeros((grid_size, grid_size, 4))  # x,y,q-value -- x,y in m
        self.origin_x = round(initial_size / 2.0 * res)
        self.origin_y = round(initial_size / 2.0 * res)

        self.learning_rate = learning_rate
        self.discount_factor = discount_rate
        self.exploration_rate = exploration_rate 
        self.res = res 

        # Add pose array publisher
        self.pose_pub = self.create_publisher(PoseArray, 'optimal_policy', 10)

        print(f"Q-table initialized with shape: {self.q_table.shape} and origin: {self.origin_x}, {self.origin_y}\n")

    def choose_action(self, state, grid):
        tried = set()
        max_actions = 4
        actions = list(range(max_actions))

        while len(tried) < max_actions:
            if random.uniform(0, 1) < self.exploration_rate:
                print("   exploring")
                action = random.choice(list(set(actions) - tried))
            else:
                print("   exploiting")
                q_vals = self.q_table[state]
                for a in np.argsort(-q_vals):
                    if a not in tried:
                        action = a
                        break
            tried.add(action)

            # Compute the next state in world coordinates and then grid indices
            state_world = grid.get_next_state(state, action)  # in meters
            next_state = grid.world_to_grid(state_world[0], state_world[1])  # in grid coords (x, y)

            # Debug info: print the grid window around considered next state
            # print(f"Trying action {action} -> next state (grid coords): {next_state}")
            # print_grid_window(grid.map, next_state, window_size=5)

            # Check that next_state is within map bounds
            if not (0 <= next_state[0] < grid.width and 0 <= next_state[1] < grid.height):
                # print(f"      Next state {next_state} out of bounds, trying next action...")
                continue

            # Check occupancy grid cell value at next_state
            cell_val = grid.map[next_state[1], next_state[0]]

            # Only allow action if cell is free (0). Block if obstacle (100) or unknown (-1)
            if cell_val == 0:
                print(f"      Chose action: {action} ({actions_list[action]}) for state: {state}")
                return action
            else:
                print(f"      Blocked by obstacle or unknown cell (val={cell_val}) at {next_state} with action {action} ({actions_list[action]}), trying next action...")

        # If all actions are blocked, default None 
        print("      All directions blocked. Returning action 'None'.")
        return None

    def update_q_value(self, state, action, reward, next_state):
        print(f"      Updating Q-value for state: {state}, action: {action}, reward: {reward}, next_state: {next_state}")
        max_future_q = np.max(self.q_table[next_state])  # Best Q-value for next state
        current_q = self.q_table[state][action]
        # Q-learning formula
        td_update = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        print(f"      Q-value update: {td_update}")
        x,y = state
        self.q_table[y][x][action] = td_update 

        print_qtable_window(self.q_table, state)
    
    def expand_qtable(self):
        current_size = self.q_table.shape[0]
        expansion_size = 10

        # create new qtable with expanded size 
        new_q_table = np.zeros((current_size+expansion_size, current_size+expansion_size, 4))
        
        # offset origin by half of expansion size 
        self.origin_x = round(self.origin_x + expansion_size/2)
        self.origin_y = round(self.origin_y + expansion_size/2)
        
        # copy old qtable to new qtable 
        new_q_table[expansion_size/2:self.q_table.shape[0], expansion_size/2:self.q_table.shape[1], :] = self.q_table
        self.q_table = new_q_table

        print(f"Q-table expanded to shape: {self.q_table.shape} and origin: {self.origin_x}, {self.origin_y}")
    
    def in_bounds(self, state):
        return 0 <= state[0] < self.q_table.shape[0] and 0 <= state[1] < self.q_table.shape[1]

    def train(self, num_episodes, grid, reward_type="manhattan"):
        print(f"<<< Training for {num_episodes} episodes >>>\n")
        for i in range(num_episodes):
            print(f"Episode {i+1}/{num_episodes}")
            state = grid.reset()
            print(f" --- State (px): {state} --- ")
            state_offset = (state[0] + self.origin_x, state[1] + self.origin_y)  # offset by origin 
            done = False

            while not done:
                action = self.choose_action(state, grid)

                next_state, reward, done = grid.step(action, reward_type)
                
                next_state_offset = (next_state[0] + self.origin_x, next_state[1] + self.origin_y)  # offset by origin 
                if not self.in_bounds(next_state_offset):
                    self.expand_qtable()
                
                self.update_q_value(state_offset, action, reward, next_state_offset)

                state = next_state
                state_offset = next_state_offset
                print(f" --- State (px): {state} --- ")

        print(f"Training complete.\n")

    def create_optimal_policy(self, grid):
        # Get start and goal states in grid coordinates
        start_x, start_y = grid.start_state
        goal_x, goal_y = grid.goal
        
        # Trace path from start to goal
        current_x, current_y = start_x, start_y
        path = [(current_x, current_y)]
        
        max_steps = 100  # Prevent infinite loops
        step_count = 0

        print("starting")
        
        while step_count < max_steps:
            print(f"Step {step_count}")
            # Get optimal action at current state
            q_values = self.q_table[current_y, current_x]
                
            optimal_action = np.argmax(q_values)
            
            # Get next state based on optimal action
            if optimal_action == 0:  # UP
                next_y = current_y + 1
                next_x = current_x
            elif optimal_action == 1:  # RIGHT
                next_y = current_y
                next_x = current_x + 1
            elif optimal_action == 2:  # DOWN
                next_y = current_y - 1
                next_x = current_x
            else:  # LEFT
                next_y = current_y
                next_x = current_x - 1
                      
            # Add to path and continue
            path.append((next_x, next_y))
            print(f"next_x: {next_x}, next_y: {next_y}")
            current_x, current_y = next_x, next_y
            
            # Check if we reached the goal
            if current_x == goal_x and current_y == goal_y: 
                break
                
            step_count += 1

        print(path)
        return path 
    
    def path_to_poses(self, path):
        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()

        # Convert path to poses
        for i in range(len(path)):
            x, y = path[i]
            # Convert grid coordinates to world coordinates offset by origin 
            world_x = x + self.origin_x
            world_y = y + self.origin_y
            
            # Create pose
            pose = Pose()
            pose.position.x = float(world_x)
            pose.position.y = float(world_y)
            pose.position.z = 0.0

            # Set orientation based on direction to next point or previous point
            if i < len(path) - 1:
                next_x, next_y = path[i + 1]
                dx = next_x - x
                dy = next_y - y
            else:
                prev_x, prev_y = path[i - 1]
                dx = x - prev_x
                dy = y - prev_y
            
            # Calculate angle based on direction
            if dx == 0:
                angle = math.pi/2 if dy < 0 else -math.pi/2
            else:
                angle = 0 if dx > 0 else math.pi
                
            q = tf_transformations.quaternion_from_euler(0, 0, angle)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            pose_array.poses.append(pose)

        # Publish the pose array
        self.pose_pub.publish(pose_array)
        print(f"Published optimal path with {len(pose_array.poses)} poses")


def main(args=None):
    rclpy.init(args=args)
        
    initial_size = 1000 
    res = 0.05

    # odom px
    startx = 0 
    starty = 0 
    goal = (1,3)  # m, odom

    gm_node = GridMapper(goal=goal, pos_x=startx, pos_y=starty, initial_size=initial_size, res=res)
    q = QLearningAgent(initial_size=initial_size, res=res)
 
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gm_node)
    executor.add_node(gm_node.planner)
    executor.add_node(q)  # Add the QLearningAgent node to the executor
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # num_episodes = 2
    # q.train(num_episodes, gm_node)
    
    # Publish the learned policy
    # print("\nPublishing optimal policy to RViz2...")
    # path = q.create_optimal_policy(gm_node)
    path = [(0,0), (0,1), (0,2), (0,3), (1,3)]
    q.path_to_poses(path)

    time.sleep(10)

if __name__ == '__main__':
    main()

