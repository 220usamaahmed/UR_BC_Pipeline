#!/bin/bash
# Starts the full stack: UR3e driver (mock or real) → MoveIt move_group → Foxglove bridge.
#
# Env vars (set defaults for mock hardware; override for the real robot):
#   ROBOT_IP           — IP of the real arm. Ignored when USE_FAKE_HARDWARE=true. (default: 192.168.56.101)
#   USE_FAKE_HARDWARE  — "true" for mock hardware, "false" to connect to ROBOT_IP. (default: true)
#   KINEMATICS_PARAMS_FILE — path to a calibration.yaml extracted from the real arm.
#                            Only meaningful when USE_FAKE_HARDWARE=false; see README for how to generate it.
set -e

source /opt/ros/humble/setup.bash

if [ -f /root/ros2_ws/install/setup.bash ]; then
  source /root/ros2_ws/install/setup.bash
fi

ROBOT_IP="${ROBOT_IP:-192.168.56.101}"
USE_FAKE_HARDWARE="${USE_FAKE_HARDWARE:-true}"

KINEMATICS_ARGS=()
if [ -n "${KINEMATICS_PARAMS_FILE:-}" ]; then
  KINEMATICS_ARGS=(kinematics_params_file:="${KINEMATICS_PARAMS_FILE}")
fi

# Step 1 — UR3e hardware driver (mock or real, per USE_FAKE_HARDWARE)
# Provides: robot_state_publisher (/tf, /robot_description), joint_state_broadcaster,
# scaled_joint_trajectory_controller.
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:="${ROBOT_IP}" \
  use_fake_hardware:="${USE_FAKE_HARDWARE}" \
  launch_rviz:=false \
  "${KINEMATICS_ARGS[@]}" &
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
