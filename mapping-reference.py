import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Twist 
import numpy as np
import math
import tf2_ros
from tf2_msgs.msg import TFMessage

# constants
DEFAULT_SCAN_TOPIC = '/scan'
DEFAULT_ODOM_TOPIC = '/odometry/filtered'
DEFAULT_MAP_TOPIC = '/map'
DEFAULT_CMD_VEL_TOPIC = '/cmd_vel'

TF_BASE_LINK = 'base_link'
TF_ODOM = 'odom'

LASER_ROBOT_OFFSET = -math.pi  # angle of the laser that is in front of the robot

class OccupancyGridMapper(Node):
    def __init__(self):
        super().__init__('occupancy_grid_mapper')
        
        # initialize parameters for the map
        self.resolution = 0.05  # meters per cell
        self.initial_size = 1000  # initial map size (cells) 
        self.map = np.full((self.initial_size, self.initial_size), -1, dtype=np.int8)  # Unknown: -1
        
        # start in center of the map
        self.origin_x = -self.initial_size * self.resolution / 2.0
        self.origin_y = -self.initial_size * self.resolution / 2.0
        self.width = self.initial_size
        self.height = self.initial_size
        
        # robot starting pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.has_pose = True
        
        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # publishers and subscribers
        self.map_pub = self.create_publisher(OccupancyGrid, DEFAULT_MAP_TOPIC, 10)
        self.laser_sub = self.create_subscription(LaserScan, DEFAULT_SCAN_TOPIC, self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, DEFAULT_ODOM_TOPIC, self.odom_callback, 10)
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        
        print('Occupancy Grid Mapper initialized')

    def odom_callback(self, msg):
        # get xy position
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # get yaw angle
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)
        self.has_pose = True

    def laser_callback(self, msg):
        
        # check need to extend the map
        grid_x, grid_y = self.world_to_grid(self.robot_x, self.robot_y)
        if not self.is_valid_cell(grid_x, grid_y):
            self.expand_map(grid_x, grid_y)
            grid_x, grid_y = self.world_to_grid(self.robot_x, self.robot_y)
        
        # process each laser scan angle
        for i, range_val in enumerate(msg.ranges):
            if math.isinf(range_val) or math.isnan(range_val):
                continue
            if range_val < msg.range_min or range_val > msg.range_max:
                continue
            
            # get angle of the laser ray
            angle = msg.angle_min + i * msg.angle_increment
 
            # get world coordinates of the laser input
            point = PointStamped()
            point.header.frame_id = msg.header.frame_id
            point.header.stamp = msg.header.stamp
            point.point.x = range_val * math.cos(angle)
            point.point.y = range_val * math.sin(angle)
            point.point.z = 0.0
            
            # transform from laser to Odom
            if self.tf_buffer.can_transform('odom', msg.header.frame_id, msg.header.stamp):  
                try:
                    transform = self.tf_buffer.lookup_transform(TF_ODOM, msg.header.frame_id, msg.header.stamp) #msg.header.stamp
                    point_odom = do_transform_point(point, transform)
                    world_x = point_odom.point.x
                    world_y = point_odom.point.y
                except Exception as e:
                    self.get_logger().warn(f'Transform failed: {str(e)}')
                    continue
            else:
                continue
            
            # convert to grid coordinates
            end_x, end_y = self.world_to_grid(world_x, world_y)
            
            # expand map if needed
            if not self.is_valid_cell(end_x, end_y):
                self.expand_map(end_x, end_y)
                end_x, end_y = self.world_to_grid(world_x, world_y)
            
            # update map using Bresenham's algorithm
            self.bresenham_line(grid_x, grid_y, end_x, end_y)
        
        self.publish_map()

    # adapted from pa3
    def world_to_grid(self, x, y):
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    # adapted from pa3
    def is_valid_cell(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    # can expand the map if the target cell is outside the current map - it will double the size of the map in the direction of the target cell
    def expand_map(self, target_x, target_y):
        print("expanding map")
        new_width = self.width
        new_height = self.height
        new_origin_x = self.origin_x
        new_origin_y = self.origin_y
        
        # determine how much to expand
        if target_x >= self.width:
            new_width = self.width * 2
        elif target_x < 0:
            new_width = self.width * 2
            new_origin_x = self.origin_x - self.width * self.resolution
        if target_y >= self.height:
            new_height = self.height * 2
        elif target_y < 0:
            new_height = self.height * 2
            new_origin_y = self.origin_y - self.height * self.resolution
        
        # new map
        new_map = np.full((new_height, new_width), -1, dtype=np.int8)
        
        # put data from old map into new map
        offset_x = int((new_width - self.width) // 2)
        offset_y = int((new_height - self.height) // 2)
        new_map[offset_y:offset_y+self.height, offset_x:offset_x+self.width] = self.map
        
        # update map
        self.map = new_map
        self.width = new_width
        self.height = new_height
        self.origin_x = new_origin_x
        self.origin_y = new_origin_y
        self.get_logger().info(f'Map expanded to {self.width}x{self.height}, origin: ({self.origin_x}, {self.origin_y})')

    # algorithm implemented from class
    def bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if self.is_valid_cell(x0, y0):
                self.map[y0, x0] = 0  # free space
            if x0 == x1 and y0 == y1:
                if self.is_valid_cell(x0, y0):
                    self.map[y0, x0] = 100  # not free
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.width
        map_msg.info.height = self.height
        map_msg.info.origin.position.x = self.origin_x
        map_msg.info.origin.position.y = self.origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        map_msg.data = self.map.flatten().tolist()
        self.map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
