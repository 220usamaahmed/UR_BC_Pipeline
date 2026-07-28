#!/bin/bash
# Starts the full stack: dependency bootstrap → colcon build → UR3e driver
# (mock or real) → MoveIt move_group → [real hardware only] suction gripper
# + ZED2i camera → Foxglove bridge.
#
# Env vars (set defaults for mock hardware; override for the real robot):
#   ROBOT_IP           — IP of the real arm. Ignored when USE_FAKE_HARDWARE=true. (default: 192.168.56.101)
#   USE_FAKE_HARDWARE  — "true" for mock hardware, "false" to connect to ROBOT_IP. (default: true)
#   KINEMATICS_PARAMS_FILE — path to a calibration.yaml extracted from the real arm.
#                            Only meaningful when USE_FAKE_HARDWARE=false; see README for how to generate it.
set -e

source /opt/ros/humble/setup.bash

# Bootstrap source dependencies into the bind-mounted workspace. Cloning at
# image-build time would not work here because /root/ros2_ws is replaced by the
# host bind mount when the container starts.
WORKSPACE_DIR="/root/ros2_ws"
SOURCE_DIR="${WORKSPACE_DIR}/src"
BOOTSTRAP_DIR="${WORKSPACE_DIR}/.bootstrap"
ROSDEP_STAMP="${BOOTSTRAP_DIR}/source_dependencies_rosdep_v1"
DEPENDENCIES_CHANGED=false

ensure_repository() {
  local repository_url="$1"
  local destination="$2"
  local package_name="$3"

  if [ -f "${destination}/package.xml" ]; then
    echo "${package_name} source already exists at ${destination}."
    return
  fi

  if [ -e "${destination}" ]; then
    echo "ERROR: ${destination} exists but is not a valid ${package_name} package." >&2
    echo "Move or remove that path so the repository can be cloned safely." >&2
    exit 1
  fi

  echo "Cloning ${package_name} into ${destination}..."
  git clone --depth 1 "${repository_url}" "${destination}"
  DEPENDENCIES_CHANGED=true
}

mkdir -p "${SOURCE_DIR}" "${BOOTSTRAP_DIR}"

ensure_repository \
  "https://github.com/AndrejOrsula/pymoveit2.git" \
  "${SOURCE_DIR}/pymoveit2" \
  "pymoveit2"

ensure_repository \
  "https://github.com/RoboticManipulation/ecpmi_gripper.git" \
  "${SOURCE_DIR}/ecpmi_gripper" \
  "ecpmi_gripper"

# rosdep resolves the package.xml dependencies. The stamp avoids an apt index
# refresh on every container restart, while a newly cloned repository always
# forces the dependency pass to run again.
if [ "${DEPENDENCIES_CHANGED}" = true ] || [ ! -f "${ROSDEP_STAMP}" ]; then
  echo "Installing source-package dependencies with rosdep..."
  apt-get update
  rosdep install \
    --from-paths "${SOURCE_DIR}/pymoveit2" "${SOURCE_DIR}/ecpmi_gripper" \
    --ignore-src \
    --rosdistro "${ROS_DISTRO:-humble}" \
    -y
  rm -rf /var/lib/apt/lists/*
  touch "${ROSDEP_STAMP}"
fi

# Step 0 — build the workspace (bc_pipeline, ecpmi_gripper, pymoveit2, ...).
# ros2_ws is bind-mounted from the host, so build/install/log persist across
# container restarts and this is an incremental (fast) build after the first run.
echo "Building ROS 2 workspace..."
cd "${WORKSPACE_DIR}"
colcon build --symlink-install
source install/setup.bash
cd /root

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

# Step 4 — Suction gripper + ZED2i camera. Both need real hardware: the
# gripper needs the UR driver's IO controller (mock hardware doesn't expose
# tool digital outputs) and the camera needs the USB/GPU passthrough that
# only docker-compose.real.yml provides. Skip both under mock hardware.
if [ "${USE_FAKE_HARDWARE}" = "false" ]; then
  echo "Real hardware detected — starting suction gripper and ZED2i camera..."

  ros2 launch ecpmi_gripper suction_gripper.launch.py &
  GRIPPER_PID=$!

  ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i &
  ZED_PID=$!
fi

# Step 5 — Foxglove bridge (foreground — keeps the container alive and the
# background processes running as its children via the process group).
exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml
