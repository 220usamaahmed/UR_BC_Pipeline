```bash

# Start docker container
./start-docker.sh --real

# Press start on robot

# Exec into container
./start-docker.sh exec bash

# Manual grip and release
ros2 run ecpmi_gripper gripper_client grip && ros2 run ecpmi_gripper gripper_client release

# Record place left (change config file for other skills; configs in ros2_ws/src/bc_pipeline/config)
ros2 launch bc_pipeline record_sequence.launch.py config:=place_left.py

# Process trajectory and visualize (change folder path for other skills)
cd /root/ros2_ws/src/processing
python3 make_training_trajectory.py /root/ros2_ws/trajectories/place_left/ --batch && python3 visualize_run.py /root/ros2_ws/trajectories/place_left/ --batch

# Run inference
ros2 run bc_pipeline inference --ros-args -p checkpoint_path:=checkpoints/...