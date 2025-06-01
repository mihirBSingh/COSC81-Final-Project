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
from rclpy.time import Time

# constants
DEFAULT_SCAN_TOPIC = '/scan'
DEFAULT_ODOM_TOPIC = '/odometry/filtered'
DEFAULT_MAP_TOPIC = '/map'
DEFAULT_CMD_VEL_TOPIC = '/cmd_vel'

TF_BASE_LINK = 'base_link'
TF_ODOM = 'odom'

LASER_ROBOT_OFFSET = -math.pi  # angle of the laser that is in front of the robot

LINEAR_VELOCITY = 0.2 # m/s
ANGULAR_VELOCITY = math.pi/6 # rad/s

USE_SIM_TIME = True 

class Mover(Node): 
    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY):
        super().__init__('mover')
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self.rate = self.create_rate(10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Workaround not to use roslaunch
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])
        
    def get_transformation(self, target_frame, start_frame, time=Time()):
            """Get transformation between two frames."""
            # print(f"   Getting transformation from {start_frame} --> {target_frame}")
            try:
                while not self.tf_buffer.can_transform(target_frame, start_frame, time): 
                    rclpy.spin_once(self)
                tf_msg = self.tf_buffer.lookup_transform(target_frame, start_frame, time)
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform: {ex}')
                return
            # self.get_logger().info(f'Received tf message: {tf_msg}')   
            translation = tf_msg.transform.translation
            quaternion = tf_msg.transform.rotation

            t = tf_transformations.translation_matrix([translation.x, translation.y, translation.z])
            R = tf_transformations.quaternion_matrix([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            T = t.dot(R)
            T = np.round(T, decimals=1) # rounding minimizes noise 
            
            # print(f"   Transformation: {T}")
            return T

    def move(self, linear_vel, angular_vel):
            """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
            # Setting velocities.
            twist_msg = Twist()

            twist_msg.linear.x = linear_vel
            twist_msg.angular.z = angular_vel
            self.cmd_pub.publish(twist_msg)

    # angles in radians
    def rotate(self,angle):
        print(f"           Rotating {angle} rad, {angle*180/math.pi} degrees")

        secs = abs(angle) / self.angular_velocity
        duration = Duration(seconds=secs)
        start_time = self.get_clock().now()

        # rotate for certain duration 
        while rclpy.ok():
            rclpy.spin_once(self)  # Process callbacks
            # Check if the specified duration has elapsed.
            if self.get_clock().now() - start_time >= duration:
                break
            # Publish the twist message continuously.
            if angle > 0: 
                self.move(0.0, self.angular_velocity)  # counterclockwise 
            else: 
                self.move(0.0, -self.angular_velocity)  # clockwise

    # distance in m 
    def translate(self,distance): 
        print(f"           Moving forward {distance}m")
        
        secs = abs(distance) / self.linear_velocity
        duration = Duration(seconds=secs)
        start_time = self.get_clock().now()

        # rotate for certain duration 
        while rclpy.ok():
            rclpy.spin_once(self)  # Process callbacks
            # Check if the specified duration has elapsed.
            if self.get_clock().now() - start_time >= duration:
                break
            # Publish the twist message continuously.
            self.move(self.linear_velocity, 0.0)  
    
    def get_angle(self, y, x):
        theta = math.atan2(y, x)
        # edge cases for tan (multiples of π/2)
        if x == 0:
            if y > 0:
                theta = math.pi/2
            else:
                theta = -math.pi/2
        return theta 
        
    def get_distance(self,p):
        return  math.sqrt((p[0])**2 + (p[1])**2)

    def move_to_point(self,x,y):
        print(f"           Moving to point: {x}, {y}")
        odom_p = np.array([x, y, 0, 1])
        bl_T2_odom = self.get_transformation(TF_BASE_LINK, TF_ODOM)
        bl_p = bl_T2_odom.dot(odom_p.transpose())
        # print(f"           bl_T2_odom: {bl_T2_odom}")
        # print(f"           Base link point: {bl_p}")

        theta = self.get_angle(bl_p[1], bl_p[0])
        dist = self.get_distance(bl_p)

        self.rotate(theta)
        self.translate(dist)

class OccupancyGridMapper(Node):
    def __init__(self, pos_x=0.0, pos_y=0.0, pos_theta=0.0, initial_size=1000, goal=(999, 999), res=0.05):
        super().__init__('occupancy_grid_mapper')
        
        # initialize parameters for the map
        self.resolution = res  # meters per cell
        self.initial_size = initial_size  # initial map size (cells) 
        self.map = np.full((self.initial_size, self.initial_size), -1, dtype=np.int8)  # Unknown: -1
        
        # start in center of the map
        self.origin_x = -self.initial_size * self.resolution / 2.0
        self.origin_y = -self.initial_size * self.resolution / 2.0
        self.width = self.initial_size
        self.height = self.initial_size
        
        # robot starting pose
        self.robot_x = pos_x
        self.robot_y = pos_y
        self.robot_theta = pos_theta
        self.has_pose = True
        
        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # publishers and subscribers
        self.map_pub = self.create_publisher(OccupancyGrid, DEFAULT_MAP_TOPIC, 10)
        self.laser_sub = self.create_subscription(LaserScan, DEFAULT_SCAN_TOPIC, self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, DEFAULT_ODOM_TOPIC, self.odom_callback, 10)
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        
        # set up qlearning states and obstacles 
        self.start_state = (self.robot_x, self.robot_y)   
        self.state = self.start_state
        self.goal = goal

        self.obstacles = set()  # TODO: detect obstacles in nearby cells
        for row in range(self.height):
            for col in range(self.width):
                value = self.map[row, col]
                if value == 100: 
                    self.obstacles.add((row, col))

        self.mover = Mover() 

        print('Occupancy Grid Mapper initialized')

    def reset(self):
        self.state = self.start_state

        # TODO move robot back to start with pathfinding using pa3 planning

        return self.state

    def is_terminal(self, state):
        return state in self.obstacles or state == self.goal
    
    def get_next_state(self, state, action):  # m
        next_state = list(state)
        
        if action == 0:  # Move up rel to odom 
            next_state[1] += STEP
        elif action == 1:  # Move right
            next_state[0] += STEP
        elif action == 2:  # Move down
            next_state[1] -= STEP
        elif action == 3:  # Move left
            next_state[0] -= STEP
        
        return tuple(next_state)
    
    def compute_reward(self, prev_state, action, type):  
        # Obstacle or goal
        state_world = self.get_next_state(prev_state, action)  # m 
        state = self.world_to_grid(state_world[0], state_world[1])  # px
        if  type == "obstacle":
            if self.map[state[0], state[1]] == 100:
                reward = -10
            elif state == self.goal:
                reward = -100
            else:
                reward = 0

        elif type == "manhattan":
                # Manhattan distance to goal
                prev_dist = abs(prev_state[0] - self.goal[0]) + abs(prev_state[1] - self.goal[1])
                new_dist = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
                reward = prev_dist - new_dist
                return reward
        print(f"        Reward: {reward}")
        return reward 
    
    def execute_action(self, action):
        print(f"        [Executing action]")
        self.mover.move_to_point(self.pos_x, self.pos_y)

    def step(self, action, reward_type):
        print(f"      [Stepping in grid]")
        next_state = self.get_next_state(self.state, action)  # m 
        state_px = self.world_to_grid(next_state[0], next_state[1])  # px
        print(f"        Next state (px): {state_px}, (m): {next_state[0]}, {next_state[1]}")
        
        reward = self.compute_reward(self.state, action, reward_type) 
        self.state = next_state
        self.pos_x, self.pos_y = next_state
        done = self.is_terminal(next_state)
        self.execute_action(action)
        rclpy.spin_once(self)  # Process any callbacks after action execution
        return next_state, reward, done

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
        print(f"        [Laser callback]")
        
        # check need to extend the map
        grid_x, grid_y = self.world_to_grid(self.robot_x, self.robot_y)
        if not self.is_valid_cell(grid_x, grid_y):
            self.expand_map(grid_x, grid_y)
            grid_x, grid_y = self.world_to_grid(self.robot_x, self.robot_y)
        
        printed = False
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
            try:
                transform = self.tf_buffer.lookup_transform(TF_ODOM, msg.header.frame_id, msg.header.stamp) #msg.header.stamp
                point_odom = do_transform_point(point, transform)
                world_x = point_odom.point.x
                world_y = point_odom.point.y
            except Exception as e:
                if not printed: 
                    self.get_logger().warn(f'Transform failed: {str(e)}')
                    printed = True
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
