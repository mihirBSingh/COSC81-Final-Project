#!/usr/bin/env python

# Author: Megan Liu
# Date: 2025/03/31

import math
import time
import numpy as np

import tf_transformations
import tf2_ros # library for transformations.
from tf2_ros import TransformException


import rclpy # module for ROS APIs
from rclpy.node import Node
# http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist 
from rclpy.duration import Duration # message type for duration

NODE_NAME = "planner"

MAP_TOPIC = "map"
POSE_TOPIC = "pose"
POSE_SEQUENCE_TOPIC = "pose_sequence"
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'

TF_BASE_LINK = 'base_link'
TF_ODOM = 'odom'
TF_MAP = 'map'

MAP_FRAME_ID = "map"

USE_SIM_TIME = True

LINEAR_VELOCITY = 0.1 # m/s
ANGULAR_VELOCITY = math.pi/15 # rad/s

class Grid:
    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.resolution = resolution
        self.height = height
        self.width  = width

    def cell_at(self, r, c):
        return self.grid[r, c]
    
    def is_valid(self, r,c): 
        r,c = int(r), int(c)
        buffer = 8

        if 0 <= r < self.height and 0 <= c < self.width and not self.grid[r-buffer:r+buffer,c-buffer:c+buffer].any():
            return True 
        
        return False

class Plan(Node):
    def __init__(self, map_frame_id=MAP_FRAME_ID, node_name=NODE_NAME, context=None):
        super().__init__(node_name, context=context)

        # Workaround not to use roslaunch
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL, 
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])

        # occgrid subscriber and info 
        self.sub = self.create_subscription(OccupancyGrid, MAP_TOPIC, self.map_callback, 1)
        self.map = None # the variable containing the map.
        self.map_frame_id = map_frame_id
        self.occgrid_frame_id = None
        

        # Setting up transformation listener.
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # pose sequence
        self.pose_seq_pub = self.create_publisher(PoseArray, POSE_SEQUENCE_TOPIC, 1) 
        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 1)

    def map_callback(self, msg):
        print("MAP CALLBACK")
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        self.occgrid_frame_id = msg.header.frame_id

    def create_pose(self, x, y, z, q):
        """Example of publishing an arrow, without orientation."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame_id
        
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        pose_msg.pose.orientation.x = float(q.x)
        pose_msg.pose.orientation.y = float(q.y)
        pose_msg.pose.orientation.z = float(q.z)
        pose_msg.pose.orientation.w = float(q.w)

        return pose_msg

    def get_transformation(self, start_frame, target_frame):
        """Get transformation between two frames."""
        try:
            while not self.tf_buffer.can_transform(target_frame, start_frame, self.get_clock().now()):
                print("waiting for transform...")
                rclpy.spin_once(self)
            tf_msg = self.tf_buffer.lookup_transform(target_frame, start_frame, self.get_clock().now())
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
        
        return T, quaternion

    # pos, orientation relative to map rf
    def get_currentloc(self):
        map_T2_bl, quaternion = self.get_transformation(TF_BASE_LINK, TF_MAP)
        print("map_T2_bl:\n", map_T2_bl)

        # position 
        bl_p = np.array([0, 0, 0, 1])
        map_p = map_T2_bl.dot(bl_p.transpose())
        pose = self.create_pose(map_p[0], map_p[1], map_p[2], quaternion)

        return pose
    
    def get_angle(self, y, x):
        theta = math.atan2(y, x)

        # edge cases for tan (multiples of π/2)
        if x == 0:
            if y > 0:
                theta = math.pi/2
            else:
                theta = -math.pi/2

        return theta 
    
    def make_quat(self, yaw):
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def px_to_pose(self, curr_px, prev_pose): 
        x, y = self.px_to_grid(curr_px[0], curr_px[1])
        px, py, _ = self.pose_to_grid(prev_pose)
        # print("         processing ", px, py, " --> ", x, y)

        # angle to rotate from prev coords --> curr coords 
        angle = self.get_angle(y-py, x-px)
        
        # print("          angle: ", angle)

        q = self.make_quat(angle)
        pose = self.create_pose(x, y, 0, q) 

        return pose

    def pose_to_grid(self, pose): 
        return (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)

    def path_to_pxposes(self, path, start_pose): # in px
        print("     converting px path to poses")
        pose_seq = [start_pose] 

        # track angle throughout poses
        quaternion = start_pose.pose.orientation 
        rpy = tf_transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        yaw = rpy[2]
        
        for i in range(1, len(path)):
            p = self.px_to_pose(path[i], pose_seq[i-1])
            pose_seq.append(p)
        
        return pose_seq
            
    def publish_posearray(self, poses_arr): 
        print("   publishing poses array")
        pa = PoseArray()

        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.map_frame_id
        pa.poses = [ps.pose for ps in poses_arr]

        self.pose_seq_pub.publish(pa)
        print("   published")
    
    def px_to_grid(self,x_w, y_w): # px to m
        # column index
        c = x_w * self.map.resolution
        # row index
        r = y_w * self.map.resolution
        
        return c, r  

    def grid_to_px(self,x, y): # m to pixels
        # column index
        r = x // self.map.resolution
        # row index
        c = y // self.map.resolution
        return r,c

    def cleanup_path(self, path): 
        # include start, end, rotations 
        clean_path = []

        # segment_start point 
        segment_start = path[0]
        matching_coord = None
        for i in range(len(path)): 
            coord = path[i]

            # if start or end, add to clean_path 
            if i == 0 or i == len(path)-1: 
                clean_path.append(path[i])
                continue 

            if matching_coord == None:
                # if x of coord matches segment_start point, 
                if path[i][0] == segment_start[0]: 
                    matching_coord = "x"
                else: 
                    matching_coord = "y"
            else:  
                x_coord_match = matching_coord == "x" and coord[0] == segment_start[0]  
                y_coord_match =  matching_coord == "y" and coord[1] == segment_start[1]

                if x_coord_match or y_coord_match: 
                    continue 
                else: 
                    clean_path.append(path[i])
                    segment_start = path[i]
                    matching_coord = None 

        return clean_path

    # on pixels 
    def treesearch(self, start, end, stack):  # with history 
        print("   TREESEARCH from ", start, " --> ", end)       
    
        visited = set()  
        prev = {start:None}  
        frontier = [start]

        while len(frontier) > 0: 
            if stack: 
                leaf = frontier.pop()
            if not stack: 
                leaf = frontier.pop(0)
            # print("leaf: ", leaf)
            visited.add(leaf)

            if leaf == end: 
                # backtrack through history to get path 
                node = end
                path = []
                while node != None: 
                    path.append(node)
                    node = prev[node]
                path.reverse()
                # print("   PX PATH::: ", path)

                return path 
            
            # add neighbors (decided 4 arbitrarily, not 8)
            x, y = leaf 
            neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]  # dfs order - down, up, left, right
            for n in neighbors: 
                c,r = n
                if self.map.is_valid(r,c) and n not in visited:  # valid bounds
                    # print("   neighbor: ", n)
                    prev[n] = leaf
                    frontier.append(n)
                    visited.add(n)

        return None 
    
    # input start pose 
    def treesearch_poses(self, start_pose, end_coords, stack): 
        # convert pose to px 
        start_coords = self.pose_to_grid(start_pose)
        start_px = self.grid_to_px(start_coords[0], start_coords[1])
        end_px = self.grid_to_px(end_coords[0], end_coords[1])

        px_path = self.treesearch(start_px, end_px, stack)
        if px_path == None: return None

        px_path = self.cleanup_path(px_path)
        print("path: ", px_path)

        # turn path into array of poses
        pose_array = self.path_to_pxposes(px_path, start_pose)
        self.publish_posearray(pose_array)
        return True 

    def bfs(self, start_pose, end_coords): 
        print("BFS  ======================")
        return self.treesearch_poses(start_pose, end_coords, False)

    def dfs(self, start_pose, end_coords): 
        print("DFS  ======================")
        return self.treesearch_poses(start_pose, end_coords, True)
    
class Execute(Node): 
    def __init__(self, node_name="execute", context=None, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY, ):
        super().__init__(node_name, context=context)

        use_sim_time = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )
        self.set_parameters([use_sim_time])
                         
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self.pose_seq_sub = self.create_subscription(PoseArray, POSE_SEQUENCE_TOPIC, self.pose_sequence_callback, 1)

        self.linear_velocity = linear_velocity # Constant linear velocity set.
        self.angular_velocity = angular_velocity # Constant angular velocity set.

        self.done = False

        self.dist = []
        self.rot = []

    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()

        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)
           
    # angles in radians
    def rotate(self, angle):
        print(f"   Rotating {angle} rad, {angle*180/math.pi} degrees")

        secs = abs(angle) / self.angular_velocity
        duration = Duration(seconds=secs)
        start_time = self.get_clock().now()

        # rotate for certain duration 
        while rclpy.ok():
            rclpy.spin_once(self)

            # Check if the specified duration has elapsed.
            if self.get_clock().now() - start_time >= duration:
                break
            # # Publish the twist message continuously.
            if angle > 0: 
                self.move(0.0, self.angular_velocity)  # counterclockwise 
            else: 
                self.move(0.0, -self.angular_velocity)  # clockwise

    # distance in m 
    def translate(self, distance): 
        print(f"   Moving forward {distance}m")
        
        secs = abs(distance) / self.linear_velocity
        duration = Duration(seconds=secs)
        start_time = self.get_clock().now()

        # rotate for certain duration 
        while rclpy.ok():
            rclpy.spin_once(self)
            # Log current time information.
            
            # Check if the specified duration has elapsed.
            if self.get_clock().now() - start_time >= duration:                
                break
            # Publish the twist message continuously.
            self.move(self.linear_velocity, 0.0)  

    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def pose_sequence_callback(self, msg): 
        print("\nPOSE SEQUENCE CALLBACK")
        poses = msg.poses

        prev_pose = None
        for p in poses:
            if prev_pose is not None:
            # Calculate distance between current pose and previous pose
                dx = p.position.x - prev_pose.position.x
                dy = p.position.y - prev_pose.position.y
                distance = math.sqrt(dx**2 + dy**2)

                # Calculate relative angle between current pose and previous pose
                prev_quaternion = prev_pose.orientation
                prev_rpy = tf_transformations.euler_from_quaternion([prev_quaternion.x, prev_quaternion.y, prev_quaternion.z, prev_quaternion.w])
                prev_yaw = prev_rpy[2]

                curr_quaternion = p.orientation
                curr_rpy = tf_transformations.euler_from_quaternion([curr_quaternion.x, curr_quaternion.y, curr_quaternion.z, curr_quaternion.w])
                curr_yaw = curr_rpy[2]

                relative_angle = curr_yaw - prev_yaw
            else:
                distance = 0  # No movement for the first pose
                relative_angle = 0  # No rotation for the first pose

            self.rot.append(relative_angle)
            self.dist.append(distance) 

            prev_pose = p

        self.done = True

    def execute(self): 
        print("Executing")
        for i in range(len(self.rot)): 
            angle = self.rot[i]
            dist = self.dist[i]

            self.rotate(angle)
            self.translate(dist)
        
        self.stop()

def main(args=None):
    # 1st. initialization of node.
    rclpy.init(args=args)

    p = Plan()
    e = Execute()

    # wait until the map callback has run
    print("Waiting for map... \n(run ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap \"{ map_url: 'pa3/maze.yml' }\" in another terminal)\n")
    
    while p.map is None:
        rclpy.spin_once(p)

    while rclpy.ok():
        print("this program runs bfs/dfs from the current location to a location of your choosing.")
        while True:
            algo = input("do you want bfs or dfs? ").strip().lower()
            if algo in ["bfs", "dfs"]:
                break
            print("invalid input - enter 'bfs' or 'dfs'")
        
        while True: 
            end_coords = input("Input the goal coordinates (format: x,y): ").strip()
            try:
                end_coords = tuple(map(int, end_coords.split(','))) + (0,)
                print(end_coords)
                if len(end_coords) == 3:
                    if not p.map.is_valid(end_coords[0], end_coords[1]):
                        print("invalid input - that value isn't in the grid") 
                        continue 
                    break
                else:
                    print("invalid input - enter exactly two numbers separated by a comma")
            except ValueError:
                print("invalid input - enter valid integers separated by a comma")
                
        start_pose = p.get_currentloc()

        print("\n")

        if algo == "bfs": 
            path_found = p.bfs(start_pose, end_coords)
        else: 
            path_found = p.dfs(start_pose, end_coords)

        if path_found is None:
            print("no path found") 
            continue 
        
        print("waiting for distance and rotation arrays to be set")
        while e.dist == [] and e.rot == []: 
            rclpy.spin_once(e)
            pass 
        e.execute()

        rclpy.spin_once(p)
        rclpy.spin_once(e)
        break 
        
    rclpy.shutdown()

if __name__ == "__main__":
    main()
