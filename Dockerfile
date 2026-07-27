FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install UR driver, ros2_control, controllers, and foxglove_bridge
RUN apt-get update && apt-get install -y \
    ros-humble-ur \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-foxglove-bridge \
    ros-humble-joint-state-publisher \
    ros-humble-moveit \
    python3-colcon-common-extensions \
    python3-pip \
    python3-pylsp \
    && rm -rf /var/lib/apt/lists/*

# Behaviour-cloning inference dependencies. The PyTorch wheel matches the
# CUDA 12.8 toolkit installed below; torchvision/torchaudio are not needed.
RUN python3 -m pip install --no-cache-dir \
        torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --no-cache-dir einops==0.8.1

# --- Stereolabs ZED2i camera: ROS deps + CUDA + ZED SDK ---
#
# The CUDA and ZED SDK installers are large (~4GB, ~2.4GB) and versioned, so
# they're kept out of git and expected to already be sitting in
# ros2_ws/downloads/ (bind-mounted host dir, gitignored) before building —
# see README for the one-time download step. Building from local files here
# (instead of `wget`-ing them in a RUN step) means a rebuild never re-fetches
# them: Docker's layer cache handles unrelated rebuilds, and there's simply no
# network call in this layer to redo even on a from-scratch build.

# ROS message/transform packages required by zed-ros2-wrapper
RUN apt-get update && apt-get install -y \
    ros-humble-geographic-msgs \
    ros-humble-cob-srvs \
    ros-humble-robot-localization \
    ros-humble-point-cloud-transport \
    wget zstd \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 toolkit (required by the ZED SDK)
COPY ros2_ws/downloads/cuda-ubuntu2204.pin /tmp/cuda-ubuntu2204.pin
COPY ros2_ws/downloads/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb /tmp/cuda-repo.deb
RUN cp /tmp/cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    dpkg -i /tmp/cuda-repo.deb && \
    cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8 libnvidia-encode-570 libnvidia-decode-570 && \
    rm -rf /var/lib/apt/lists/* /tmp/cuda-repo.deb /var/cuda-repo-ubuntu2204-12-8-local

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${PATH}:${CUDA_HOME}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64

# ZED SDK 5.0
COPY ros2_ws/downloads/ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run /tmp/zed_sdk_installer.run
RUN chmod +x /tmp/zed_sdk_installer.run && \
    /tmp/zed_sdk_installer.run -- silent skip_cuda && \
    rm -f /tmp/zed_sdk_installer.run && \
    chown -R root:root /usr/local/zed && \
    chmod -R a+rx /usr/local/zed && \
    chmod -R 755 /usr/local/zed/resources

ENV ZED_SDK_ROOT=/usr/local/zed

# Pre-fetch this camera's factory calibration file so the SDK doesn't need
# network access on first open. Pass --build-arg ZED_CAMERA_SERIAL=<SN> if a
# different physical camera is used (find it with `ZED_Explorer -a`). Best
# effort only — if there's no network at build time the SDK will just fetch
# it itself on first camera open instead.
ARG ZED_CAMERA_SERIAL=35477861
RUN mkdir -p /root/.zed /usr/local/zed/settings && \
    ( wget -q --timeout=10 --tries=2 \
        "https://www.stereolabs.com/developers/calib/?SN=${ZED_CAMERA_SERIAL}" \
        -O /root/.zed/SN${ZED_CAMERA_SERIAL}.conf \
      && cp /root/.zed/SN${ZED_CAMERA_SERIAL}.conf /usr/local/zed/settings/SN${ZED_CAMERA_SERIAL}.conf \
      && chmod 644 /root/.zed/SN${ZED_CAMERA_SERIAL}.conf /usr/local/zed/settings/SN${ZED_CAMERA_SERIAL}.conf \
    ) || echo "WARNING: could not pre-fetch calibration for SN${ZED_CAMERA_SERIAL}, will fetch at runtime instead"

# Source ROS automatically in every shell session
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Startup script that launches the UR3e mock hardware + Foxglove bridge.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /root
