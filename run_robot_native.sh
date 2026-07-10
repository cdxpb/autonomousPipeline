#!/bin/bash
# Native (no-Docker) launcher. Run ON the Duckiebot itself, after
# setup_native_jetson.sh has completed all stages and the workspace is built.
#
# Usage:
#   bash run_robot_native.sh --approach 7
#   bash run_robot_native.sh --approach 5 --backend onnx    # fallback/A-B testing
#   bash run_robot_native.sh --approach 7 --model-set full  # older 480x640 models
#   bash run_robot_native.sh --approach 7 --no-telemetry
#   bash run_robot_native.sh --approach 5 --telemetry-every-n 5 --model-set full
#   bash run_robot_native.sh --approach 5 --yolo-backend tensorrt --pilotnet-backend onnx
#   bash run_robot_native.sh --approach 7 --smoothing 0.3   # EMA-smooth published velocity (1.0 = off)

set -e

APPROACH=7
BACKEND=tensorrt
YOLO_BACKEND=""
PILOTNET_BACKEND=""
MODEL_SET=smaller
TELEMETRY=true
TELEMETRY_EVERY_N=1
SMOOTHING=1.0
WS_DIR="$HOME/native_ws"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach)           APPROACH="$2";          shift ;;
        --backend)            BACKEND="$2";            shift ;;
        --yolo-backend)       YOLO_BACKEND="$2";       shift ;;
        --pilotnet-backend)   PILOTNET_BACKEND="$2";   shift ;;
        --model-set)          MODEL_SET="$2";          shift ;;
        --no-telemetry)       TELEMETRY=false ;;
        --telemetry-every-n)  TELEMETRY_EVERY_N="$2";  shift ;;
        --smoothing)          SMOOTHING="$2";          shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$YOLO_BACKEND" ]; then YOLO_BACKEND="$BACKEND"; fi
if [ -z "$PILOTNET_BACKEND" ]; then PILOTNET_BACKEND="$BACKEND"; fi

export VEHICLE_NAME=duckiexp
export ROS_MASTER_URI=http://localhost:11311
# not 127.0.0.1, must be reachable by remote subscribers (the Mac dashboard)
export ROS_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
if [ -z "$ROS_IP" ]; then
    echo "WARNING: could not auto-detect this robot's LAN IP; falling back to 127.0.0.1 (remote subscribers won't be able to connect)"
    export ROS_IP=127.0.0.1
fi

echo "=== Duckiebot native inference launcher ==="
echo "Approach: $APPROACH | YOLO: $YOLO_BACKEND | PilotNet: $PILOTNET_BACKEND | Model set: $MODEL_SET"
echo "Telemetry: $TELEMETRY | Smoothing: $SMOOTHING"
echo "ROS_MASTER_URI: $ROS_MASTER_URI (expects the Docker roscore container to be --net host)"
echo "ROS_IP: $ROS_IP"

source /opt/ros/melodic/setup.bash
source "$WS_DIR/devel/setup.bash"

# 30 is the camera driver's native default (confirmed via rostopic hz); setting 20 once
# broke the driver outright and needed a container restart.
rosparam set /$VEHICLE_NAME/camera_node/framerate 30

if [ "$MODEL_SET" == "full" ]; then
    YOLO_ENGINE="/data/models/yolo_model/yolo_model.engine"
    if [ "$APPROACH" == "5" ]; then
        PILOTNET_ENGINE="/data/models/pilotnet/best_model_regNheadv2.engine"
    else
        PILOTNET_ENGINE="/data/models/pilotnet/best_model_approach7.engine"
    fi
else
    YOLO_ENGINE="/data/models/yolo_model/yolo_best_256x320.engine"
    if [ "$APPROACH" == "5" ]; then
        PILOTNET_ENGINE="/data/models/pilotnet/best_model_approach5_smaller.engine"
    else
        PILOTNET_ENGINE="/data/models/pilotnet/best_model_approach7_smaller.engine"
    fi
fi

if [ "$YOLO_BACKEND" == "tensorrt" ] && [ ! -f "$YOLO_ENGINE" ]; then
    echo "NOTE: YOLO .engine not found yet. Build it first:"
    echo "  python3 $HOME/dev/autonomousPipeline/packages/cond_imitation_learning_pkg/src/export_trt.py --yolo --model-set $MODEL_SET"
fi
if [ "$PILOTNET_BACKEND" == "tensorrt" ] && [ ! -f "$PILOTNET_ENGINE" ]; then
    echo "NOTE: PilotNet .engine not found yet. Build it first:"
    echo "  python3 $HOME/dev/autonomousPipeline/packages/cond_imitation_learning_pkg/src/export_trt.py --approach $APPROACH --model-set $MODEL_SET"
fi

rosrun cond_imitation_learning_pkg headless_autonomous_node.py \
    --approach "$APPROACH" \
    _yolo_backend:="$YOLO_BACKEND" \
    _pilotnet_backend:="$PILOTNET_BACKEND" \
    _model_set:="$MODEL_SET" \
    _publish_telemetry:="$TELEMETRY" \
    _telemetry_every_n:="$TELEMETRY_EVERY_N" \
    _smoothing_alpha:="$SMOOTHING"
