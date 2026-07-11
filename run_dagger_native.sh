#!/bin/bash
# Native Mac DAgger data collection: runs dagger_node.py (AI + correction logging) and the
# data_collector_pkg UI (WASD drive/record/intent controls) against the robot's camera+wheels,
# no Docker/robot-side compute needed. Approach 5/7 only (see dagger_node.py PILOTNET_PATH).
#
# Usage:
#   bash run_dagger_native.sh                       # approach 5, onnx backend, ~/Desktop/my_dataset
#   bash run_dagger_native.sh --approach 7
#   bash run_dagger_native.sh --approach 5 --data-dir ~/Desktop/dagger_run3
#   bash run_dagger_native.sh --approach 7 --fwd-speed 0.35 --turn-bias 0.2
#   bash run_dagger_native.sh --backend pytorch                          # fallback/A-B testing
#   bash run_dagger_native.sh --yolo-backend onnx --pilotnet-backend pytorch
set -e

APPROACH=5
DATA_DIR="$HOME/Desktop/my_dataset"
FWD_SPEED=0.3
TURN_BIAS=0.15
BACKEND=onnx
YOLO_BACKEND=""
PILOTNET_BACKEND=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach)           APPROACH="$2";           shift ;;
        --data-dir)           DATA_DIR="$2";            shift ;;
        --fwd-speed)          FWD_SPEED="$2";           shift ;;
        --turn-bias)          TURN_BIAS="$2";           shift ;;
        --backend)            BACKEND="$2";             shift ;;
        --yolo-backend)       YOLO_BACKEND="$2";        shift ;;
        --pilotnet-backend)   PILOTNET_BACKEND="$2";    shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$YOLO_BACKEND" ]; then YOLO_BACKEND="$BACKEND"; fi
if [ -z "$PILOTNET_BACKEND" ]; then PILOTNET_BACKEND="$BACKEND"; fi

if [[ "$APPROACH" != "5" && "$APPROACH" != "7" ]]; then
    echo "ERROR: --approach must be 5 or 7 (got $APPROACH)"; exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="$HOME/dev/native_ws"

export ROS_MASTER_URI=http://duckiexp.local:11311
export ROS_IP=$(ipconfig getifaddr en0)
export VEHICLE_NAME=duckiexp

echo "=== Native Mac DAgger collection ==="
echo "Approach: $APPROACH | YOLO: $YOLO_BACKEND | PilotNet: $PILOTNET_BACKEND | Data dir: $DATA_DIR | fwd_speed: $FWD_SPEED | turn_bias: $TURN_BIAS"
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "ROS_IP: $ROS_IP"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ros_native

# First-run setup: wire dagger_pkg/data_collector_pkg into the native workspace (mirrors how
# cond_imitation_learning_pkg is already symlinked in) and build if either is missing.
NEED_BUILD=0
for pkg in dagger_pkg data_collector_pkg; do
    if [ ! -e "$WS_DIR/src/$pkg" ]; then
        ln -s "$REPO_DIR/packages/$pkg" "$WS_DIR/src/$pkg"
        echo "Linked $pkg into $WS_DIR/src/"
        NEED_BUILD=1
    fi
done
if [ $NEED_BUILD -eq 1 ]; then
    echo "=== Building dagger_pkg + data_collector_pkg (first run only) ==="
    (cd "$WS_DIR" && catkin build dagger_pkg data_collector_pkg)
fi

source "$WS_DIR/devel/setup.bash"
mkdir -p "$DATA_DIR"

echo ""
echo "Controls (focus the 'CIL Data Collector & Tuner' window):"
echo "  Drive:   W/A/S/D"
echo "  Record:  R (toggles correction logging on/off -- only logs while you're driving)"
echo "  Model:   M (toggles the AI driving autonomously; DAgger = correct it live with WASD)"
echo "  Intent:  I (straight) J (left) L (right) K (lane_following)"
echo ""

rosrun dagger_pkg dagger_node.py --approach "$APPROACH" --data-dir "$DATA_DIR" --fwd-speed "$FWD_SPEED" --turn-bias "$TURN_BIAS" \
    --yolo-backend "$YOLO_BACKEND" --pilotnet-backend "$PILOTNET_BACKEND" &
DAGGER_PID=$!
trap "echo 'Stopping dagger_node...'; kill $DAGGER_PID 2>/dev/null" EXIT INT TERM

rosrun data_collector_pkg ui_node.py
