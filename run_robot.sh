#!/bin/bash
# Launcher that runs INSIDE the Duckiebot's Docker container.
#
# Usage (from inside `dts devel run -H duckiexp.local --cmd bash`):
#   bash /code/catkin_ws/src/autonomousPipeline/run_robot.sh --approach 7
#
# Or directly via dts:
#   dts devel run -H duckiexp.local --cmd "bash /code/catkin_ws/src/autonomousPipeline/run_robot.sh --approach 7 --backend onnx"

APPROACH=7
BACKEND=onnx

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach) APPROACH="$2"; shift ;;
        --backend)  BACKEND="$2";  shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "=== Duckiebot headless inference launcher (approach=$APPROACH, backend=$BACKEND) ==="

echo "[1/3] Re-building catkin workspace..."
source /opt/ros/noetic/setup.sh
catkin build --workspace /code/catkin_ws/ --no-status -q
source /code/catkin_ws/devel/setup.bash

if [ ! -d "/data/models/pilotnet" ]; then
    echo
    echo "ERROR: Models not found at /data/models/pilotnet/"
    echo "Transfer models from your Mac first:"
    echo "  rsync -av models/pilotnet/best_model_approach7.onnx  duckie@duckiexp.local:/data/models/pilotnet/"
    echo "  rsync -av models/pilotnet/best_model_regNheadv2.onnx duckie@duckiexp.local:/data/models/pilotnet/"
    echo "  rsync -av models/yolo_model/yolo_model.onnx          duckie@duckiexp.local:/data/models/yolo_model/"
    exit 1
fi

echo "[2/3] Models OK"
echo "[3/3] Starting headless_autonomous_node..."

exec rosrun cond_imitation_learning_pkg headless_autonomous_node.py \
    --approach "$APPROACH" \
    _backend:="$BACKEND"
