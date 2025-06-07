# COSC81-Final-Project

## Overview

In this project, we implemented two reinforcement learning algorithms for target finding using ROSBot2: Q-learning and value iteration.
We initially buit a simple static custom map for training and testing in Gazebo simulator but then decided to go with Stage simulator (and a maze map from PA3) since training was faster.
Below are instructions to train the two algorithms.

## QLearning

1.  start up docker container

    `docker compose up`

2.  kill docker gazebo simulator (we're using stage)

3.  for each of the following commands, open a new terminal, run `docker compose exec ros bash`, then the command

    `ros2 launch stage_ros2 stage.launch.py world:=/root/catkin_ws/src/cs81-finalproj/maze enforce_prefixes:=false one_tf_tree:=true` - launches stage simulator

    `ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 1 map rosbot/odom` - publishes the static transform from map and odom

    `python3 qlearning.py` - run qlearning.py

    `rviz2` - run rviz2 to visualize the occupancy grid

    <!-- `ros2 run teleop_twist_keyboard teleop_twist_keyboard` - teleoperate/drive the rosbot in the simulation -->

## Value Iteration

## Submodule Instructions

Megan forked the VNC-ROS repo by AQL and wanted to use the docker container/easy access to previous code for the final project. Mihir created a repo for our final project, which I made into a submodule: a repo inside a repo. In the parent repo, the child repo is a gitlink (a pointer to a specific commit). If you cd into the submodule folder, it looks like the remote branch. Any commits made within that folder update only that child repo, and any commits made outside that folder update only the parent repo.
Here's how to set up.

1. Within workspace/src, run

`git submodule add https://github.com/mihirBSingh/COSC81-Final-Project.git cs81-finalproj`

2. Check it worked by running

`git submodule status`

In Cursor/VSCode with the Git version control extension, a green "A" should appear next to the submodule, denoting a new submodule added to the index. After changes in the submodule, there should be a blue "S" denoting submodule having staged changes.
