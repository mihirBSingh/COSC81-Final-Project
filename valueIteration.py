#!/usr/bin/env python

# Author: Mihir Singh
# Date: 2025/05/10

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
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry, OccupancyGrid


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
# actions - these represent movement in block coordinates
ACTIONS = {
    'N': (1, 0),  # North: up in grid
    'S': (-1, 0),   # South: down in grid  
    'E': (0, 1),   # East: right in grid
    'W': (0, -1),  # West: left in grid
    'Stay': (0, 0)
}

# probabilities for actions (deterministic for now)
DIRECTION_PROBABILITIES = {
    'N': [('N', 1.0)],
    'S': [('S', 1.0)],
    'E': [('E', 1.0)], 
    'W': [('W', 1.0)],
    'Stay': [('Stay', 1.0)]
}

MOVE_DISTANCE_THRESHOLD = 0.1  # How close to center before considering "arrived"
STATE_MOVE_TIMEOUT = 10.0  

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
        buffer_radius = 0
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
        # publishers
        self.policy_pub = self.create_publisher(MarkerArray, 'policy_visualization', 10)
        self.value_pub = self.create_publisher(MarkerArray, 'value_visualization', 10)

        # robot constants
        self.linear_velocity = LINEAR_VELOCITY
        self.angular_velocity = ANGULAR_VELOCITY
        # init constants
        self.map = None
        self.map_frame_id = map_frame_id
        self.map_origin = None
        self.occupancy_grid = None
        
        # Fixed grid division - 20x20 blocks
        self.grid_rows = 20
        self.grid_cols = 20
        
        self._odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # State-based navigation variables
        self.current_pose = None
        self.current_state = (2,2)  # Current (block_r, block_c)
        self.target_state = None   # Target (block_r, block_c) 
        self.robot_policy = None
        self.goal_state = None
        self.navigation_active = False
        self.moving_to_state = False
        self.state_start_time = None
        
        # Control timer - faster for responsive movement
        self.control_timer = self.create_timer(0.1, self.state_based_control_loop)
        
    # get and update map
    def map_callback(self, msg):
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        self.map_origin = msg.info.origin
        self.occupancy_grid = msg
        print(f"Got map: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}")
    
    def odom_callback(self, msg):
        print("Received odometry update")
        self.current_pose = msg.pose.pose

        if self.map is not None and self.map_origin is not None:
            # Get robot position (assuming odom aligns with map frame for now)
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            
            # Convert world coordinates to state (block coordinates)
            self.current_state = self.world_to_grid(x, y)
            
            # Debug output
            if hasattr(self, '_last_logged_state'):
                if self._last_logged_state != self.current_state:
                    self.get_logger().info(f"Robot moved to state: {self.current_state}")
                    self._last_logged_state = self.current_state
            else:
                self._last_logged_state = self.current_state
    
    def get_state_center_world(self, state_r: int, state_c: int) -> Tuple[float, float]:
        return self.grid_to_world(state_r, state_c)

    def get_current_robot_state(self) -> Tuple[int, int]:
        """Get the current state the robot is in"""
        return self.current_state
        
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
    
    def get_states(self):
        # Force 10x10 state space
        return [(block_r, block_c) for block_r in range(20) for block_c in range(20)]

    def get_rewards(self, goal: Tuple[int, int]):
        rewards = {}
        block_size_x = math.ceil(self.map.width / 20)
        block_size_y = math.ceil(self.map.height / 20)
        goal_block = (goal[0] // block_size_y, goal[1] // block_size_x)

        for block_r in range(20):
            for block_c in range(20):
                # Calculate cell range for this block
                start_r = block_r * block_size_y
                end_r = min((block_r + 1) * block_size_y, self.map.height)
                start_c = block_c * block_size_x
                end_c = min((block_c + 1) * block_size_x, self.map.width)
                
                # Check if any cell in the block is invalid
                is_block_valid = True
                for r in range(start_r, end_r):
                    for c in range(start_c, end_c):
                        if not self.map.is_valid(r, c):
                            is_block_valid = False
                            break
                    if not is_block_valid:
                        break
                
                # Assign rewards
                if (block_r, block_c) == goal_block and is_block_valid:
                    rewards[(block_r, block_c)] = 10
                elif not is_block_valid:
                    rewards[(block_r, block_c)] = -1  # Reduced penalty
                else:
                    rewards[(block_r, block_c)] = -0.1
        return rewards

    def grid_to_world(self, block_r: int, block_c: int) -> Tuple[float, float]:
        block_size_x = math.ceil(self.map.width / 20)
        block_size_y = math.ceil(self.map.height / 20)
        # Center of the block in grid coordinates
        center_r = block_r * block_size_y + block_size_y // 2
        center_c = block_c * block_size_x + block_size_x // 2
        center_r = min(center_r, self.map.height - 1)
        center_c = min(center_c, self.map.width - 1)
        x = center_c * self.map.resolution + self.map_origin.position.x + self.map.resolution / 2
        y = center_r * self.map.resolution + self.map_origin.position.y + self.map.resolution / 2
        return x, y

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        block_size_x = math.ceil(self.map.width / 20)
        block_size_y = math.ceil(self.map.height / 20)
        x_rel = x - self.map_origin.position.x
        y_rel = y - self.map_origin.position.y
        col = int(x_rel / (self.map.resolution * block_size_x))
        row = int(y_rel / (self.map.resolution * block_size_y))
        col = min(max(col, 0), 19)  # Clamp to 0-19
        row = min(max(row, 0), 19)
        return row, col

    def get_block_size(self):
        """Calculate block size based on map dimensions"""
        block_height = self.map.height // self.grid_rows
        block_width = self.map.width // self.grid_cols
        return block_height, block_width

    def is_block_valid(self, block_r: int, block_c: int) -> bool:
        """Check if a block contains any obstacles"""
        block_height, block_width = self.get_block_size()
        
        # Calculate pixel boundaries for this block
        start_row = block_r * block_height
        end_row = min((block_r + 1) * block_height, self.map.height)
        start_col = block_c * block_width  
        end_col = min((block_c + 1) * block_width, self.map.width)
        
        # Check all pixels in the block
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                if not self.map.is_valid(r, c):
                    return False
        return True

    def get_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Get next state given current state and action"""
        move = ACTIONS[action]
        next_r = state[0] + move[0]
        next_c = state[1] + move[1]
        
        # Check bounds
        if 0 <= next_r < self.grid_rows and 0 <= next_c < self.grid_cols:
            return (next_r, next_c)
        else:
            return state  # Stay in place if out of bounds

    def value_iteration(self, start: Tuple[int, int], goal: Tuple[int, int], discount_factor: float = 0.9, theta: float = 0.01):
        block_size = 1
        start_block = (start[0] // block_size, start[1] // block_size)
        goal_block = (goal[0] // block_size, goal[1] // block_size)
        states = self.get_states()
        rewards = self.get_rewards(goal)

        if not states:
            self.get_logger().error("No valid states found in the grid!")
            return {}, {}

        V = {state: 0.0 for state in states}
        policy = {state: None for state in states}
        max_iterations = 20000
        iteration = 0

        while True:
            delta = 0.0
            for state in states:
                if state == goal_block:
                    V[state] = rewards.get(state, 10)  # Terminal state
                    policy[state] = 'Stay'
                    continue
                old_v = V[state]
                action_values = {}
                for action, move in ACTIONS.items():
                    value = 0
                    for direction, prob in DIRECTION_PROBABILITIES[action]:
                        next_block_r, next_block_c = state[0] + move[0], state[1] + move[1]
                        next_state = (next_block_r, next_block_c)
                        # Check if next state is within bounds and valid
                        if (0 <= next_block_r < (self.map.height + block_size - 1) // block_size and
                            0 <= next_block_c < (self.map.width + block_size - 1) // block_size and
                            self.map.is_valid(next_block_r * block_size, next_block_c * block_size)):
                            value += prob * (rewards.get(next_state, -1) + discount_factor * V.get(next_state, 0))
                        else:
                            value += prob * (rewards.get(state, -1) + discount_factor * V.get(state, 0))
                    action_values[action] = value
                V[state] = max(action_values.values())
                policy[state] = max(action_values, key=action_values.get)
                delta = max(delta, abs(old_v - V[state]))
            iteration += 1
            self.get_logger().info(f"Iteration {iteration}: delta = {delta}")
            if delta < theta:
                self.get_logger().info(f"Converged after {iteration} iterations with delta = {delta}")
                break
            if iteration >= max_iterations:
                self.get_logger().warning(f"Stopped after {max_iterations} iterations without convergence (delta = {delta})")
                break

        return V, policy

    def publish_policy_visualization(self, policy, V):
        marker_array = MarkerArray()
        block_size_x = math.ceil(self.map.width / 20)
        block_size_y = math.ceil(self.map.height / 20)
        for i, (state, action) in enumerate(policy.items()):
            if action is None or action == 'Stay':
                continue
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "policy_arrows"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            block_r, block_c = state
            center_r = block_r * block_size_y + block_size_y // 2
            center_c = block_c * block_size_x + block_size_x // 2
            center_r = min(center_r, self.map.height - 1)
            center_c = min(center_c, self.map.width - 1)
            world_x, world_y = self.grid_to_world(block_r, block_c)
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.1
            if action == 'N':
                yaw = np.pi/2
            elif action == 'S':
                yaw = -np.pi/2
            elif action == 'E':
                yaw = 0.0
            elif action == 'W':
                yaw = np.pi
            else:
                continue
            quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]
            marker.scale.x = min(block_size_x, block_size_y) * self.map.resolution * 0.8
            marker.scale.y = min(block_size_x, block_size_y) * self.map.resolution * 0.2
            marker.scale.z = min(block_size_x, block_size_y) * self.map.resolution * 0.2
            value = V.get(state, 0.0)
            max_value = max(V.values()) if V.values() else 1.0
            min_value = min(V.values()) if V.values() else 0.0
            normalized_value = 0.5 if max_value == min_value else (value - min_value) / (max_value - min_value)
            marker.color.r = 1.0 - normalized_value
            marker.color.g = normalized_value
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)
        self.policy_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} policy arrows")

    def publish_value_visualization(self, V):
        marker_array = MarkerArray()
        block_size_x = math.ceil(self.map.width / 20)
        block_size_y = math.ceil(self.map.height / 20)
        max_value = max(V.values()) if V.values() else 1.0
        min_value = min(V.values()) if V.values() else 0.0
        for i, (state, value) in enumerate(V.items()):
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "value_function"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            block_r, block_c = state
            world_x, world_y = self.grid_to_world(block_r, block_c)
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            cube_size = min(block_size_x, block_size_y) * self.map.resolution * 0.9
            marker.scale.x = cube_size
            marker.scale.y = cube_size
            marker.scale.z = 0.02
            normalized_value = 0.5 if max_value == min_value else (value - min_value) / (max_value - min_value)
            marker.color.r = 1.0 - normalized_value
            marker.color.g = 0.0
            marker.color.b = normalized_value
            marker.color.a = 0.6
            marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)
        self.value_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} value cubes")
        
    def state_based_control_loop(self):
        if not self.navigation_active or self.current_state is None or self.robot_policy is None:
            return
        print("State-based control loop running")
        # Check if we've reached the goal state
        if self.current_state == self.goal_state:
            self.get_logger().info(f"SUCCESS: Reached goal state {self.goal_state}!")
            self.stop_robot()
            self.navigation_active = False
            return
        
        # If we're not currently moving between states, decide next move
        if not self.moving_to_state:
            self.decide_next_state_move()
        else:
            # Continue moving toward target state
            self.execute_state_movement()

    def decide_next_state_move(self):
        current_policy_action = self.robot_policy.get(self.current_state, 'Stay')
        
        # self.get_logger().info(f"Current state: {self.current_state}, Policy action: {current_policy_action}")
        
        if current_policy_action == 'Stay':
            self.get_logger().info("Policy says Stay - stopping")
            self.stop_robot()
            return
        
        # Calculate target state based on policy action
        self.target_state = self.get_next_state(self.current_state, current_policy_action)
        
        # Validate target state is different and valid
        if self.target_state == self.current_state:
            self.get_logger().warn("Target state same as current state")
            self.stop_robot()
            return
        
        if not self.is_state_valid(self.target_state):
            # self.get_logger().error(f"Target state {self.target_state} is invalid!")
            self.stop_robot()
            return
        
        # Start moving to target state
        self.get_logger().info(f"Moving from state {self.current_state} to {self.target_state} (action: {current_policy_action})")
        self.moving_to_state = True
        self.state_start_time = self.get_clock().now()

    def execute_state_movement(self):
        if self.current_pose is None:
            print("No current pose available, cannot execute movement")
        if self.target_state is None or self.current_pose is None:
            print("found")
            return
        
        # Get world coordinates of target state center
        target_x, target_y = self.get_state_center_world(self.target_state[0], self.target_state[1])
        
        # Calculate distance and angle to target
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # Get current robot orientation
        current_quat = self.current_pose.orientation
        current_euler = tf_transformations.euler_from_quaternion([
            current_quat.x, current_quat.y, current_quat.z, current_quat.w
        ])
        current_yaw = current_euler[2]
        
        # Calculate angle difference
        angle_diff = target_angle - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2*math.pi
        while angle_diff < -math.pi:
            angle_diff += 2*math.pi
        
        # Create movement command
        cmd = Twist()
        
        # Check if we've arrived at target state
        if distance < MOVE_DISTANCE_THRESHOLD:
            self.get_logger().info(f"Arrived at target state {self.target_state}")
            self.stop_robot()
            self.moving_to_state = False
            self.target_state = None
            return
        
        # Check for timeout
        # if self.state_start_time is not None:
        #     elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        #     if elapsed > STATE_MOVE_TIMEOUT:
        #         self.get_logger().warn(f"Timeout moving to state {self.target_state}, stopping")
        #         self.stop_robot()
        #         self.moving_to_state = False
        #         return
        
        # Movement control: turn first, then move
        if abs(angle_diff) > math.pi / 4:
            # Need to turn toward target
            cmd.angular.z = self.angular_velocity if angle_diff > 0 else -self.angular_velocity
            cmd.linear.x = 0.0
        else:
            # Move forward toward target
            cmd.linear.x = self.linear_velocity
            # Small angular correction
            cmd.angular.z = 0.3 * angle_diff
        
        self.cmd_vel_pub.publish(cmd)

    def is_state_valid(self, state: Tuple[int, int]) -> bool:
        state_r, state_c = state
        if state_r < 0 or state_r >= 20 or state_c < 0 or state_c >= 20:
            return False
        return self.is_block_valid(state_r, state_c)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
    def start_state_navigation(self, goal_state: Tuple[int, int]):
        if self.robot_policy is None:
            self.get_logger().error("No policy computed yet!")
            return False
        
        if self.current_state is None:
            self.get_logger().error("Don't know current robot state!")
            return False
        
        self.goal_state = goal_state
        self.navigation_active = True
        self.moving_to_state = False
        
        self.get_logger().info(f"Starting navigation from state {self.current_state} to goal state {self.goal_state}")
        return True

    def compute_policy_and_navigate(self, goal_state: Tuple[int, int]):
        """Compute policy for current position to goal, then start navigation"""
        if self.current_state is None:
            self.get_logger().error("Don't know current robot position!")
            return False
        
        # Convert states back to grid coordinates for value iteration
        start_grid = self.current_state
        goal_grid = goal_state
        
        self.get_logger().info(f"Computing policy from {start_grid} to {goal_grid}")
        
        # Run value iteration
        V, policy = self.value_iteration(start_grid, goal_grid)
        
        if policy:
            self.robot_policy = policy
            
            # Publish visualizations
            self.publish_policy_visualization(policy, V)
            self.publish_value_visualization(V)
            
            # Start state-based navigation
            return self.start_state_navigation(goal_state)
        else:
            self.get_logger().error("Failed to compute policy!")
            return False
            
    def spin(self):
        start = (2, 2)  # Block coordinates (0-19)
        goal = (6, 6)   # Block coordinates (0-19)
        print("Starting Value Iteration...")
        V, policy = self.value_iteration(start, goal)
        print("Value Function:")
        for state, value in V.items():
            print(f"Block {state}: {value}")
        print("\nPolicy:")
        for state, action in policy.items():
            print(f"Block {state}: {action}")
        self.publish_policy_visualization(policy, V)
        self.publish_value_visualization(V)
        with open('policy.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State', 'Action'])
            for state, action in policy.items():
                writer.writerow([str(state), action])

        # Set goal state 
        goal_state = (3, 4)  # State coordinates (0-19, 0-19)
        
        self.get_logger().info(f"Goal state: {goal_state}")
        
        # Compute policy and start navigation
        self.current_state = (2, 2)  # Reset to initial state
        if self.compute_policy_and_navigate(goal_state):
            self.get_logger().info("State-based navigation started!")
            
            # Keep spinning until navigation completes
            while self.navigation_active and rclpy.ok():
                self.state_based_control_loop()
                rclpy.spin_once(self, timeout_sec=0.1)
        else:
            self.get_logger().error("Failed to start navigation!")
        

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
