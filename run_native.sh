#!/bin/bash
# ---------------------------------------------------------
# Native Mac ROS Runner
# ---------------------------------------------------------

# Exit if any command fails
set -e

# Setup ROS networking
export ROS_MASTER_URI=http://duckiexp.local:11311
# Dynamically get the Mac's IP address on the Wi-Fi interface (en0)
export ROS_IP=$(ipconfig getifaddr en0)
export VEHICLE_NAME=duckiexp

echo "==========================================="
echo " Starting Native ROS Noetic on Apple Silicon"
echo " ROS_MASTER_URI: $ROS_MASTER_URI"
echo " ROS_IP: $ROS_IP"
echo "==========================================="

# Activate Conda and Source Catkin Workspace
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ros_native
source ~/dev/native_ws/devel/setup.bash

# Run the node natively
rosrun cond_imitation_learning_pkg autonomous_node.py --approach 7
