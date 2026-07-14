#!/bin/bash
# Usage:
#   bash run_native_vlm.sh                 # approach 5 (default)
#   bash run_native_vlm.sh --approach 7
set -e

APPROACH=7

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach) APPROACH="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

export ROS_MASTER_URI=http://duckiexp.local:11311
export ROS_IP=$(ipconfig getifaddr en0)
export VEHICLE_NAME=duckiexp

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "ROS_IP: $ROS_IP"
echo "Approach: $APPROACH"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ros_native
source ~/dev/native_ws/devel/setup.bash

python3 -c "import transformers" 2>/dev/null || echo "NOTE: 'transformers' not found in ros_native -- the VLM panel will fail to load until you 'pip install transformers' in this env (CIL/PilotNet still work without it)."

rosrun cond_imitation_learning_pkg vlm_classifier_node.py --approach "$APPROACH"
