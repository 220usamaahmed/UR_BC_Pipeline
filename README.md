# UR_Sim — Behavior-Cloning Data Pipeline

A ROS 2 pipeline to record scripted demonstrations on a UR3e manipulator and process them into behavior-cloning training datasets.

## Setup

### Start/stop Docker container
```bash
docker compose up -d --build
docker compose down
```

### Shell into container
```bash
docker exec -it ur3e_sim bash
```

## Building & running

### Build the pipeline
```bash
cd /root/ros2_ws
colcon build --packages-select bc_pipeline --symlink-install
source install/setup.bash
```

### Record a demonstration
Run with recording and obstacles:
```bash
ros2 launch bc_pipeline record_sequence.launch.py config:=drawer_demo.yaml
```

Run without recording:
```bash
cd /root/ros2_ws
ros2 run bc_pipeline sequence_runner --ros-args -p config:=src/bc_pipeline/config/drawer_demo.yaml
```

## Processing

### Generate training trajectory from a run
```bash
cd /root/ros2_ws/src/processing
python3 make_training_trajectory.py /root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```

### Run dummy inference (policy stand-in)
```bash
ros2 run bc_pipeline dummy_inference --ros-args \
  -p run:=/root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```
