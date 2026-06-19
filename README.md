# ROS 2 + UR3e Mock Hardware on macOS

This repo runs the Universal Robots driver with **mock hardware** inside a
Docker container (amd64). A `docker-compose.yml` builds the image, starts the
UR3e simulator and the Foxglove bridge automatically, and mounts a workspace
folder so you can write ROS 2 nodes on your host and run them in the container.

## Start everything

```bash
docker compose up --build
```

This brings up:
- the UR3e driver with `use_fake_hardware:=true`
- the Foxglove bridge on `ws://localhost:8765`

The `./ros2_ws` folder on your host is mounted at `/root/ros2_ws` in the
container.

To stop:

```bash
docker compose down
```

## Write and run your own nodes

Put your packages under [`ros2_ws/src`](ros2_ws/src). An example package
`ur_demo` is already included with a dummy controller.

Open a shell in the running container:

```bash
docker compose exec ur_sim bash
```

### Option A — run a Python node directly (fastest, no build)

```bash
cd /root/ros2_ws
python3 src/ur_demo/ur_demo/dummy_controller.py
```

### Option B — build the colcon workspace and use `ros2 run`

```bash
cd /root/ros2_ws
colcon build
source install/setup.bash
ros2 run ur_demo dummy_controller
```

The dummy controller sends a single `FollowJointTrajectory` goal moving the arm
to a target pose — the Python equivalent of the manual command below.

## Manual reference commands

These are run from a shell inside the container
(`docker compose exec ur_sim bash`). The compose `command` already launches the
first two for you.

### Start the UR3e with mock hardware

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.56.101 \
  use_fake_hardware:=true \
  launch_rviz:=false
```

### Start Foxglove Bridge

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

### Send commands to the arm

```bash
ros2 action send_goal /scaled_joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "{
    trajectory: {
      joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
                    wrist_1_joint, wrist_2_joint, wrist_3_joint],
      points: [
        {
          positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
          time_from_start: {sec: 5, nanosec: 0}
        }
      ]
    }
  }"
```

### Switch controllers

```bash
# List all controllers and their state
ros2 control list_controllers

# Switch from trajectory controller to forward position controller
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
```
