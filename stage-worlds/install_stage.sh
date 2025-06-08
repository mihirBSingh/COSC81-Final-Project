source /opt/ros/${ROS_DISTRO}/setup.bash  && \
    apt -y update && apt-get install -y \
    git cmake g++ libjpeg8-dev libpng-dev libglu1-mesa-dev libltdl-dev libfltk1.1-dev ros-${ROS_DISTRO}-ament-cmake \
    && cd /root/catkin_ws/src && \
    git clone --branch ros2 https://github.com/tuw-robotics/Stage.git && \
    git clone --branch humble https://github.com/tuw-robotics/stage_ros2.git && \
    cd .. && \
    colcon build --symlink-install --cmake-args -DOpenGL_GL_PREFERENCE=LEGACY && \
    colcon build --symlink-install --packages-select stage_ros2 
