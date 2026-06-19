FROM ros:humble-ros-base

# Install UR driver, ros2_control, controllers, and foxglove_bridge
RUN apt-get update && apt-get install -y \
    ros-humble-ur \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-foxglove-bridge \
    ros-humble-joint-state-publisher \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Source ROS automatically in every shell session
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

WORKDIR /root
