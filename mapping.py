#!/usr/bin/env python

# mapper.py, building on top of pa4_mapping, used for Stage
# Author: Scottie Yang
# Date: May 20, 2025

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Twist 
import numpy as np
import math
import tf2_ros

DEFAULT_CMD_VEL_TOPIC = '/cmd_vel'
DEFAULT_SCAN_TOPIC = '/scan'

DEFAULT_MAP_TOPIC = '/map'
DEFAULT_ODOM_TOPIC = '/odometry/filtered'

TF_BASE_LINK = 'base_link'
TF_ODOM = 'odom'

LASER_ROBOT_OFFSET = -math.pi

class GridMapper(Node):
    def __init__(self):

        super().__init__('grid_mapper')

        self.res = 0.05  # m/cell
        self.initial_size = 1000  # cells
        self.map = np.full((self.initial_size, self.initial_size), -1, dtype=np.int8)
        
        self.origin_x = -self.initial_size * self.res / 2.0
        self.origin_y = -self.initial_size * self.res / 2.0
        self.width = self.initial_size
        self.height = self.initial_size
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_theta = 0.0
        self.has_pose = True
        
        # set up TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # set up publishers and subscribers
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self.odom_sub = self.create_subscription(Odometry, DEFAULT_ODOM_TOPIC, self.get_curr_pose, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, DEFAULT_MAP_TOPIC, 10)
        self.laser_sub = self.create_subscription(LaserScan, DEFAULT_SCAN_TOPIC, self.laser_callback, 10)
        
        print("OG Mapper initialized.")

    def get_curr_pose(self, msg):

        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.pos_theta = math.atan2(siny_cosp, cosy_cosp)
        self.has_pose = True

    def world_to_grid(self, x, y):
        return int((x - self.origin_x) / self.res), int((y - self.origin_y) / self.res)

    def valid_cell(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def bresenham(self, x0, y0, x1, y1): # refactored Bresenham from lec
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy
        
        while True:
            if self.valid_cell(x0, y0):
                self.map[y0, x0] = 0  # free space
            if x0 == x1 and y0 == y1:
                if self.valid_cell(x0, y0):
                    self.map[y0, x0] = 100  # not free
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def expand_map(self, target_x, target_y):
        
        self.get_logger().info("Expanding map...")
        width_new = self.width * 2
        height_new = self.height * 2

        origin_x_new = self.origin_x
        if target_x < 0:
            origin_x_new = self.origin_x - self.width * self.res

        origin_y_new = self.origin_y
        if target_y < 0:
            origin_y_new = self.origin_y - self.height * self.res
        
        map_new = np.full((height_new, width_new), -1, dtype=np.int8) # populate new map
        offset_x = int((width_new - self.width) // 2)
        offset_y = int((height_new - self.height) // 2)
        map_new[offset_y:offset_y+self.height, offset_x:offset_x+self.width] = self.map
        
        self.map = map_new # update map
        self.width = width_new
        self.height = height_new
        self.origin_x = origin_x_new
        self.origin_y = origin_y_new

        self.get_logger().info(f'Map expanded to {self.width}x{self.height}, origin: ({self.origin_x}, {self.origin_y})')

    def laser_callback(self, msg):
    
        if not self.has_pose: 
            return 
        
        grid_x, grid_y = self.world_to_grid(self.pos_x, self.pos_y)
        if not self.valid_cell(grid_x, grid_y): # extend the map first if not valid
            self.expand_map(grid_x, grid_y) 
            grid_x, grid_y = self.world_to_grid(self.pos_x, self.pos_y)
        
        for i, range in enumerate(msg.ranges): # laser scan angle

            if not (msg.range_min <= range <= msg.range_max):
                continue
        
            angle = msg.angle_min + i * msg.angle_increment # angle processing
 
            laser_point = PointStamped() # world coordinate processing
            laser_point.header.frame_id = msg.header.frame_id
            laser_point.header.stamp = msg.header.stamp
            laser_point.point.x = range * math.cos(angle)
            laser_point.point.y = range * math.sin(angle)
            
            if self.tf_buffer.can_transform('odom', msg.header.frame_id, msg.header.stamp): # transform to odom
                try:
                    transform = self.tf_buffer.lookup_transform(TF_ODOM, msg.header.frame_id, msg.header.stamp) #msg.header.stamp
                    point_odom = do_transform_point(laser_point, transform)
                    world_x = point_odom.point.x
                    world_y = point_odom.point.y
                except Exception as e:
                    self.get_logger().warn(f'Transform failed: {str(e)}')
                    continue
            else:
                continue
            
            end_x, end_y = self.world_to_grid(world_x, world_y) # export to grid
            if not self.valid_cell(end_x, end_y):
                self.expand_map(end_x, end_y)
                end_x, end_y = self.world_to_grid(world_x, world_y)

            self.bresenham(grid_x, grid_y, end_x, end_y) # map update using Bresenham
        
        self.publish_map()

    def publish_map(self):

        map_publish_msg = OccupancyGrid()

        now = self.get_clock().now().to_msg()
        map_publish_msg.header.stamp = now
        map_publish_msg.header.frame_id = 'map'

        map_publish_msg.info.resolution = self.res
        map_publish_msg.info.width = self.width
        map_publish_msg.info.height = self.height
        map_publish_msg.info.origin.position.x = self.origin_x
        map_publish_msg.info.origin.position.y = self.origin_y
        map_publish_msg.info.origin.position.z = 0.0
        map_publish_msg.info.origin.orientation.w = 1.0

        map_publish_msg.data = self.map.flatten().tolist()
        self.map_pub.publish(map_publish_msg)

def main(args=None):
    
    rclpy.init(args=args)
    node = GridMapper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()