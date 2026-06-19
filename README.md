# Build container
```bash
docker build --platform linux/amd64 -t ros2_ur3e .
```

# Start container
```bash
./start.sh
```

# Start the UR3e with Mock Hardware
```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e \
  robot_ip:=192.168.56.101 \
  use_fake_hardware:=true \
  launch_rviz:=false
```

# Start Foxglove Bridge
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

# Send Commands to the Arm
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

# Switch controllers
```bash
# List all controllers and their state
ros2 control list_controllers

# Switch from trajectory controller to forward position controller
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
```
