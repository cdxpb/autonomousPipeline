#!/bin/bash
# ============================================================
# run_robot_native.sh – Native (no-Docker) launcher, run ON the
# Duckiebot itself after setup_native_jetson.sh has completed
# all stages and the workspace has been built.
#
# Usage (on the robot):
#   bash run_robot_native.sh --approach 7
#   bash run_robot_native.sh --approach 5 --backend onnx   # fallback/A-B testing
# ============================================================

set -e

APPROACH=7
BACKEND=tensorrt
WS_DIR="$HOME/native_ws"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach) APPROACH="$2"; shift ;;
        --backend)  BACKEND="$2";  shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

export VEHICLE_NAME=duckiexp
export ROS_MASTER_URI=http://localhost:11311
# NOT 127.0.0.1: ROS advertises this address to OTHER hosts (like the Mac dashboard)
# for pulling topic data. Loopback means something different on each machine, so a
# remote subscriber would try to connect back to itself instead of this robot.
export ROS_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
if [ -z "$ROS_IP" ]; then
    echo "WARNING: could not auto-detect this robot's LAN IP; falling back to 127.0.0.1 (remote subscribers, e.g. the Mac dashboard, won't be able to connect)"
    export ROS_IP=127.0.0.1
fi

echo "======================================================="
echo " Duckiebot Native (no-Docker) Inference Launcher"
echo " Approach : $APPROACH"
echo " Backend  : $BACKEND"
echo " ROS_MASTER_URI: $ROS_MASTER_URI (expects the Docker roscore container to be --net host)
 ROS_IP: $ROS_IP"
echo "======================================================="

source /opt/ros/melodic/setup.bash
source "$WS_DIR/devel/setup.bash"

# FPS uncapped for now (camera driver at its native rate, node's own throttle
# effectively disabled) to minimize latency while debugging. Re-cap later with:
#   rosparam set /$VEHICLE_NAME/camera_node/framerate 15
# and _target_fps:=15 on the rosrun line below.

if [ "$APPROACH" == "5" ]; then
    PILOTNET_ENGINE="/data/models/pilotnet/best_model_regNheadv2.engine"
else
    PILOTNET_ENGINE="/data/models/pilotnet/best_model_approach7.engine"
fi

if [ "$BACKEND" == "tensorrt" ] && { [ ! -f "/data/models/yolo_model/yolo_model.engine" ] || [ ! -f "$PILOTNET_ENGINE" ]; }; then
    echo "NOTE: .engine files not found yet. Build them first:"
    echo "  python3 $HOME/dev/autonomousPipeline/packages/cond_imitation_learning_pkg/src/export_trt.py --all"
fi

rosrun cond_imitation_learning_pkg headless_autonomous_node.py \
    --approach "$APPROACH" \
    _backend:="$BACKEND"
