#!/bin/bash
# Native Mac ROS runner: full local-inference dashboard, no robot needed.
set -e

export ROS_MASTER_URI=http://duckiexp.local:11311
export ROS_IP=$(ipconfig getifaddr en0)
export VEHICLE_NAME=duckiexp

echo "=== Native ROS Noetic on Apple Silicon ==="
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "ROS_IP: $ROS_IP"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ros_native
source ~/dev/native_ws/devel/setup.bash

rosrun cond_imitation_learning_pkg autonomous_node.py --approach 5
