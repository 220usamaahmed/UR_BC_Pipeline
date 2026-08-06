# UR_Sim — Behavior-Cloning Data Pipeline

A ROS 2 pipeline to record scripted demonstrations on a UR3e manipulator and process them into behavior-cloning training datasets.

## Setup

### Quick start (using the start script)

The `start-docker.sh` script handles container startup with support for both mock and real hardware modes, plus automatic external drive mounting.

**Start the container (mock hardware by default):**
```bash
chmod +x start-docker.sh
./start-docker.sh
```

**Start with real hardware:**
```bash
./start-docker.sh --real
```

**Shell into the running container:**
```bash
./start-docker.sh exec bash
```

**Run a command in the container:**
```bash
./start-docker.sh exec colcon build --packages-select bc_pipeline
```

**External drive (optional):**

If an external drive exists at `/media/siddiquieu1/AHMED/new-ur3e-trajectories` on the host, it will automatically be mounted at `/data/external` inside the container. The script logs whether it found the drive or not when starting.

**Stop the container:**
Press `Ctrl+C` in the terminal where you ran `./start-docker.sh`, then:
```bash
docker compose down
```

### ROS 2 domain isolation

The complete stack (UR driver, MoveIt, gripper, ZED camera, and Foxglove
bridge) uses ROS domain ID `1` by default. Interactive shells opened with
`docker compose exec` inherit the same value, so commands such as
`ros2 topic list` see this stack without additional setup.

To intentionally use a different domain, set the same value for every ROS 2
participant that must communicate:

```bash
ROS_DOMAIN_ID=42 ./start-docker.sh --real
```

ROS domain IDs separate DDS discovery traffic but are not authentication or
encryption. Another process configured with the same domain ID can discover
the topics.

### Manual Docker commands (alternative)

If you prefer to run docker directly without the script:

```bash
# Mock hardware (default)
docker compose up --build

# Real hardware
docker compose -f docker-compose.yml -f docker-compose.real.yml up --build

# Shell into running container
docker compose exec -it ur_sim bash

# Stop
docker compose down
```

## Mock vs. real hardware

By default `docker-compose.yml` brings up the UR3e **mock** driver
(`use_fake_hardware:=true`) — no physical robot needed.

**Using the start script (recommended):**
```bash
./start-docker.sh          # mock hardware (default)
./start-docker.sh --real   # real hardware
```

**Manual setup (if not using the script):**

To drive the **real** robot, layer `docker-compose.real.yml` on top and
provide the robot's IP. Copy `.env.example` to `.env` and fill in your robot's 
details — Docker Compose loads `.env` automatically:

```bash
cp .env.example .env   # then edit ROBOT_IP (and KINEMATICS_PARAMS_FILE once you have one)
docker compose -f docker-compose.yml -f docker-compose.real.yml up --build
```

`.env` is gitignored since it's machine/robot-specific. You can also skip it
and pass the vars inline for a one-off:

```bash
ROBOT_IP=192.168.1.10 docker compose -f docker-compose.yml -f docker-compose.real.yml up --build
```

This overrides `USE_FAKE_HARDWARE` to `false`, sets `network_mode: host` (the
driver's RTDE/TCP connection needs to reach the robot's subnet directly — the
container's own bridged network and the `ports:` mapping don't apply here;
Foxglove is reachable on the host's `8765` instead), and requires `ROBOT_IP`
to be set explicitly so nobody drives the real arm by accident.

Prerequisites on the physical setup (not handled by this repo):
- The External Control URCap installed and running on the teach pendant.
- The robot in Remote Control mode.
- The e-stop chain wired and verified.

### Calibration (real robot only)

Each physical UR arm has small factory-specific kinematic offsets. The mock
driver doesn't care, but for the real robot you should extract its calibration
once and feed it back into the driver — otherwise reported TCP/joint poses can
be off by a few millimeters, which will show up as error in anything recorded
or replayed.

Extract it once (from a machine that can reach the robot, e.g. inside the
container with `ROBOT_IP` pointed at the arm):

```bash
ros2 run ur_calibration ur_calibration_node \
  --robot_ip 192.168.1.10 \
  --output_file /root/ros2_ws/calibration/ur3e_calibration.yaml
```

Save the output under `ros2_ws/` (it's bind-mounted, so it persists on the
host and is reusable across container rebuilds) and point the real-hardware
run at it:

```bash
ROBOT_IP=192.168.1.10 \
KINEMATICS_PARAMS_FILE=/root/ros2_ws/calibration/ur3e_calibration.yaml \
docker compose -f docker-compose.yml -f docker-compose.real.yml up --build
```

Re-run the extraction if you ever swap which physical arm you're driving.

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

**Process multiple runs at once:**
```bash
# Batch mode: scans all subdirectories and processes runs without training_trajectory.npz
python3 make_training_trajectory.py /root/ros2_ws/runs --batch
```

### Visualize a run
```bash
cd /root/ros2_ws/src/processing
python3 visualize_run.py /root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```

Saves a video (or animated GIF if ffmpeg isn't available) showing depth frames
alongside joint position traces with a moving cursor marking the current time.

**Process multiple runs at once:**
```bash
# Batch mode: visualizes all runs with training_trajectory.npz but no visualization yet
python3 visualize_run.py /root/ros2_ws/runs --batch

# With custom sampling (render every 10th sample instead of every 20th)
python3 visualize_run.py /root/ros2_ws/runs --batch --draw_every 10
```

### Run dummy inference (policy stand-in)
```bash
ros2 run bc_pipeline dummy_inference --ros-args \
  -p run:=/root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```

---

# Stereolabs ZED2i camera

CUDA 12.8 and the ZED SDK 5.0 are built into the Docker image (see
`Dockerfile`) — you don't need to install anything by hand inside the
container. The camera needs real USB hardware, so it's wired up through the
same `docker-compose.real.yml` overlay used for the real UR arm:

```bash
docker compose -f docker-compose.yml -f docker-compose.real.yml up --build
```

That overlay adds `privileged: true` plus the `/dev`, `/run/udev`, and GPU
passthrough the camera needs. Without `privileged`, camera opening fails fast
with `CAMERA STREAM FAILED TO START` — the ZED SDK needs to manage USB
power-management (autosuspend) around its USB3 video stream, which an
unprivileged container's read-only `/sys` doesn't allow.

### One-time setup: download the installers

The CUDA and ZED SDK installers are large (~4GB, ~2.4GB) and versioned, so
they're kept out of git. Download them once into `ros2_ws/downloads/`
(bind-mounted, gitignored — persists across rebuilds, never re-downloaded):

```bash
mkdir -p ros2_ws/downloads && cd ros2_ws/downloads

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
wget "https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu22" -O ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run
```

`docker compose ... up --build` then bakes both into the image. The
Dockerfile also pre-fetches this specific camera's factory calibration file
(serial `35477861`) at build time; pass `--build-arg
ZED_CAMERA_SERIAL=<SN>` (from `ZED_Explorer -a`) if you're using a different
physical camera.

### Build the ROS 2 packages

`zed-ros2-wrapper` (and its `zed-ros2-interfaces` submodule) live under
`ros2_ws/src/` like `bc_pipeline` — bind-mounted, built inside the container:

```bash
cd /root/ros2_ws
rosdep update && rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Run it

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

---

# ECPMi suction gripper

`ecpmi_gripper` (https://github.com/RoboticManipulation/ecpmi_gripper) controls
the ECPMi suction gripper mounted on the UR3e's tool flange through the UR
driver's IO controller (Tool Digital Output 1 by default). It lives under
`ros2_ws/src/` like the other external packages here — bind-mounted, built
inside the container. Its only deps (`rclpy`, `ur_msgs`) are already covered
by the `ros-humble-ur` install in the Dockerfile, so no image changes were
needed.

### Build

```bash
cd /root/ros2_ws
colcon build --packages-select ecpmi_gripper --symlink-install
source install/setup.bash
```

### Run it

Requires the real robot (`docker-compose.real.yml`) with the UR driver's IO
controller active — the mock driver doesn't expose tool digital outputs.

```bash
# Start the service
ros2 launch ecpmi_gripper suction_gripper.launch.py

# Command it
ros2 run ecpmi_gripper gripper_client grip     # or: release, blow
```

## Visualzing point clouds

The camera seems to be flipped left right

```bash
ros2 run tf2_ros static_transform_publisher -0.05 1.0 0.5 1.571 0.0 0.0 base_link zed_left_camera_frame
```


  ros2 launch ur_moveit_config ur_moveit.launch.py \
    ur_type:=ur3e \
    launch_servo:=true \
    launch_rviz:=true