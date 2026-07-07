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

## Mock vs. real hardware

By default `docker-compose.yml` brings up the UR3e **mock** driver
(`use_fake_hardware:=true`) — no physical robot needed.

To drive the **real** robot instead, layer `docker-compose.real.yml` on top and
provide the robot's IP. Easiest way: copy `.env.example` to `.env` and fill in
your robot's details — Docker Compose loads `.env` automatically, so you don't
need to pass anything on the command line:

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

### Run dummy inference (policy stand-in)
```bash
ros2 run bc_pipeline dummy_inference --ros-args \
  -p run:=/root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```

---

# ROS 2 packages for using Stereolabs ZED Camera cameras

```bash
```


















```bash

sudo apt update
sudo apt install wget

sudo apt install zstd

apt-get update && apt-get install -y libblas-dev liblapack-dev


Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
You might want to run 'apt --fix-broken install' to correct these.
The following packages have unmet dependencies:
 libnvidia-decode : Depends: libnvidia-compute (= 610.43.02-1ubuntu1)
 nvidia-driver : Depends: libnvidia-gl (= 610.43.02-1ubuntu1)
                 Depends: libnvidia-compute (= 610.43.02-1ubuntu1)
                 Depends: xserver-xorg-video-nvidia (= 610.43.02-1ubuntu1) but it is not going to be installed
E: Unmet dependencies. Try 'apt --fix-broken install' with no packages (or specify a solution).

apt --fix-broken install


wget https://download.stereolabs.com/zedsdk/4.1/cu121/ubuntu22 -O ZED_SDK_Linux_Ubuntu22.run
chmod +x ZED_SDK_Linux_Ubuntu22.run
./ZED_SDK_Linux_Ubuntu22.run -- silent


cd ~/ros2_ws/src/ #use your current ros2 workspace folder
git clone https://github.com/stereolabs/zed-ros2-wrapper.git
cd ..
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y # install dependencies
colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc) # build the workspace
echo source $(pwd)/install/local_setup.bash >> ~/.bashrc # automatically source the installation in every new bash (optional)
source ~/.bashrc
```