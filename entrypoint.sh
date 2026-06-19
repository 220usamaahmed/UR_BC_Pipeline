#!/bin/bash
# Starts the UR3e mock-hardware driver and the Foxglove bridge.
set -e

source /opt/ros/humble/setup.bash

# If the workspace has been built, source it too so installed nodes are available.
if [ -f /root/ros2_ws/install/setup.bash ]; then
  source /root/ros2_ws/install/setup.bash
fi

# Launch the UR3e driver with mock hardware in the background.
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.56.101 \
  use_fake_hardware:=true \
  launch_rviz:=false &

# Launch the Foxglove bridge in the foreground (keeps the container alive).
exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml
