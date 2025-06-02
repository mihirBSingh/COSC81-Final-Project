#!/usr/bin/env python

# Author: Rebecca Liu
# Date: 2025/05/04

import numpy as np
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
import tf_transformations
import tf2_ros
from tf2_ros import TransformException


NODE_NAME = "planner"
MAP_TOPIC = "map"
POSE_TOPIC = "pose_sequence"
TF_BASE_LINK = 'rosbot/base_link'
MAP_FRAME_ID = "map"
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
USE_SIM_TIME = True

LINEAR_VELOCITY = 0.1  # m/s
ANGULAR_VELOCITY = math.pi / 4  # rad/s
FREQUENCY = 10  # Hz
ROBOT_RADIUS = 0.25

#grid class to represent map
class Grid:
    def __init__(self, occupancy_grid_data, width, height, resolution, origin):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin

        #call function to expand the obstocales on the grid data
        self.expand_obstacles(ROBOT_RADIUS)

    def is_free(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] == 0

    #transform world to grid coord
    def world_to_grid(self, x, y):
        col = int((x - self.origin.x) / self.resolution)
        row = int((y - self.origin.y) / self.resolution)
        return (row, col)

    #transform grid to world grid
    def grid_to_world(self, row, col):
        x = col * self.resolution + self.origin.x + self.resolution / 2.0
        y = row * self.resolution + self.origin.y + self.resolution / 2.0
        return (x, y)

    #expand obstacle points on grid maps using robot radius
    def expand_obstacles(self, r):
        expanded_r = int(math.ceil(r / self.resolution))
        expanded_grid = self.grid.copy()

        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] != 0:
                    for i in range(-expanded_r, expanded_r + 1):
                        for j in range(-expanded_r, expanded_r + 1):
                            nr, nc = r + i, c + j
                            if 0 <= nr < self.height and 0 <= nc < self.width:
                                distance = math.sqrt(i**2 + j**2) * self.resolution
                                if distance <= r:
                                    #set to 100 for cells in r radius of obstacles
                                    expanded_grid[nr, nc] = 100  

        #update grid
        self.grid = expanded_grid



#DFS and BFS searh function
def plan_path(grid, start_pos, goal_pos, algorithm="BFS"):

    #calculate start cell in grid coordinates
    start_cell = grid.world_to_grid(*start_pos)
    goal_cell = grid.world_to_grid(*goal_pos)

    frontier = deque() if algorithm == "BFS" else []
    came_from = {}
    visited = set()
    frontier.append(start_cell)
    visited.add(start_cell)


    #search algorithm
    while frontier:
        current = frontier.popleft() if algorithm == "BFS" else frontier.pop()

        if current == goal_cell:
            break

        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + d_row, current[1] + d_col)
            if grid.is_free(*neighbor) and neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
                came_from[neighbor] = current

    if goal_cell not in came_from:
        return []

    #reconstruct path
    path = []
    current = goal_cell
    while current != start_cell:
        path.append(current)
        current = came_from[current]
    path.append(start_cell)
    path.reverse()

    poses = []
    for i in range(len(path)):
        x, y = grid.grid_to_world(*path[i])
        pose = Pose()
        pose.position.x = x
        pose.position.y = y 

        #calculate angle and x and y positions in world coordinates
        if i < len(path) - 1:
            x2, y2 = grid.grid_to_world(*path[i + 1])
            yaw = math.atan2(y2 - y, x2 - x)
        else:
            x2, y2 = grid.grid_to_world(*path[i - 1])
            yaw = math.atan2(y - y2, x - x2)

        #append pose to pose list
        q = tf_transformations.quaternion_from_euler(0, 0, yaw, 'rxyz')
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        poses.append(pose)

    return poses


#node to follow the pose sequence
class Plan(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )])

        #setting up publishers and subscribers
        self.sub = self.create_subscription(OccupancyGrid, MAP_TOPIC, self.map_callback, 1)
        self.pose_pub = self.create_publisher(PoseArray, POSE_TOPIC, 1)
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)

        self.map = None
        self.map_msg = None
        self.start = None
        self.goal = None

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def map_callback(self, msg):
        self.get_logger().info("Received OccupancyGrid.")
        self.map_msg = msg
        self.map = Grid(
            msg.data,
            msg.info.width,
            msg.info.height,
            msg.info.resolution,
            msg.info.origin.position
        )

    def move(self, linear_vel, angular_vel):
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)

    def stop(self):
        self._cmd_pub.publish(Twist())

    def spin(self, angle):
        self.stop()
        rotate_sec = abs(angle) / ANGULAR_VELOCITY
        angular_vel = ANGULAR_VELOCITY if angle > 0 else -ANGULAR_VELOCITY

        duration = Duration(seconds=rotate_sec)
        start_time = self.get_clock().now()

        while rclpy.ok():
            rclpy.spin_once(self)
            self.move(0.0, angular_vel)
            if self.get_clock().now() - start_time >= duration:
                break

        self.stop()

    def drive_straight(self, distance):
        duration = Duration(seconds=abs(distance) / LINEAR_VELOCITY)
        start_time = self.get_clock().now()

        while rclpy.ok():
            rclpy.spin_once(self)
            self.move(LINEAR_VELOCITY, 0.0)
            if self.get_clock().now() - start_time >= duration:
                break

        self.stop()

    #following the pose sequence
    def follow_path(self, pose_sequence):
        for pose in pose_sequence:
            try:
                tf_msg = self.tf_buffer.lookup_transform(MAP_FRAME_ID, TF_BASE_LINK, rclpy.time.Time())
                current_x = tf_msg.transform.translation.x
                current_y = tf_msg.transform.translation.y
                q = tf_msg.transform.rotation
                _, _, current_angle = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            except TransformException as ex:
                self.get_logger().info(f"Transform error: {ex}")

            target_x = pose.position.x
            target_y = pose.position.y
            target_angle = math.atan2(target_y - current_y, target_x - current_x)
            angle_diff = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi

            self.spin(angle_diff)

            dist = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
            self.drive_straight(dist)

    #find the pose sequence, publish the pose sequence, and then follow it
    def path_follower(self, x,y,algo):
        #print("in path follower")
        if self.map is None:
            #print("no map")
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(MAP_FRAME_ID, TF_BASE_LINK, rclpy.time.Time())
            self.start = (tf_msg.transform.translation.x, tf_msg.transform.translation.y)
        except TransformException as ex:
            self.get_logger().warn(f"Transform error: {ex}")
            return

        goal_x = x
        goal_y = y

        self.goal = (float(goal_x), float(goal_y) )

        #switch algorithm here between BFS and DFS
        poses = plan_path(self.map, self.start, self.goal, algo)


        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.map_msg.header.frame_id
        pose_array.poses = poses

        self.pose_pub.publish(pose_array)
        self.get_logger().info(f"Published path with {len(poses)} poses.")

        #follow the path after its been published
        self.follow_path(poses)


def main(args=None):
    rclpy.init(args=args)
    print("starting pa3")
    p = Plan()

    while rclpy.ok():
        goal_x = input("Enter goal x:")
        goal_y = input("Enter goal y:")
        algo = input("Enter algorithm (BFS or DFS):")
        p.path_follower(goal_x, goal_y, algo)
        rclpy.spin_once(p)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
