# Start/Stop docker
```bash
docker compose up -d --build
```

# Exec into container
```bash
docker exec -it ur3e_sim bash 
```

# Build code
```bash
colcon build --packages-select bc_pipeline --symlink-install
source install/setup.bash
```

# Run controller without recording
```bash
cd ros2_ws
ros2 run bc_pipeline sequence_runner --ros-args -p config:=src/bc_pipeline/config/drawer_demo.yaml
```

# Run with recording and obstacles
```bash
ros2 launch bc_pipeline record_sequence.launch.py config:=drawer_demo.yaml
```

# Process ROS2 Bag
```bash
python3 make_training_trajectory.py /root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```

# Run dummy inference
```bash
ros2 run bc_pipeline dummy_inference --ros-args \
  -p run:=/root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```
