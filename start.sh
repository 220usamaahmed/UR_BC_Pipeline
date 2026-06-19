#!/bin/bash
# Starts the ROS 2 container with ports exposed for Foxglove
docker run -it --rm \
  --name ur3e_sim \
  --platform linux/amd64 \
  -p 8765:8765 \
  ros2_ur3e \
  bash
