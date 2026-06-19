#!/bin/bash
# Starts the full stack: UR3e mock driver → MoveIt move_group → Foxglove bridge.
set -e

source /opt/ros/humble/setup.bash

if [ -f /root/ros2_ws/install/setup.bash ]; then
  source /root/ros2_ws/install/setup.bash
fi

# Step 1 — UR3e mock hardware driver
# Provides: robot_state_publisher (/tf, /robot_description), joint_state_broadcaster,
# scaled_joint_trajectory_controller.
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.56.101 \
  use_fake_hardware:=true \
  launch_rviz:=false &
UR_PID=$!

# Step 2 — Wait for the driver's joint_state_broadcaster to publish /joint_states.
# move_group reads the current robot state at startup; without joint states it
# enters a degraded state and its action server may not work correctly.
echo "Waiting for UR driver to initialise..."
sleep 8

# Step 3 — MoveIt move_group
# Provides: /move_group action server (collision-aware planning + execution).
# launch_servo:=false  — servo_node competes with the trajectory controller; disable it.
# launch_rviz:=false   — no display in the container.
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur3e \
  launch_rviz:=false \
  launch_servo:=false &
MG_PID=$!

# Step 4 — Foxglove bridge (foreground — keeps the container alive and the
# background processes running as its children via the process group).
exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml
