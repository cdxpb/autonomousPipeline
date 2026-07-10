#!/bin/bash
# Native Mac ROS runner: VLM dashboard. Starts vlm_server.py in the background, then
# the VLM-enabled dashboard UI. Expects headless_autonomous_node.py already running on
# the Jetson (see run_robot_native.sh) publishing telemetry.
#
# Usage:
#   bash run_dashboard_vlm.sh                          # auto-starts vlm_server.py (default)
#   bash run_dashboard_vlm.sh --local                  # no server; VLMOracle runs in-process
#   bash run_dashboard_vlm.sh --model HuggingFaceTB/SmolVLM2-500M-Instruct --precision fp16
#   bash run_dashboard_vlm.sh --port 8001

set -e

LOCAL=false
MODEL="HuggingFaceTB/SmolVLM2-256M-Instruct"
PRECISION="fp16"
PORT=8000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local)     LOCAL=true ;;
        --model)     MODEL="$2";     shift ;;
        --precision) PRECISION="$2"; shift ;;
        --port)      PORT="$2";      shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_SERVER="$SCRIPT_DIR/packages/cond_imitation_learning_pkg/src/vlm_server.py"

export ROS_MASTER_URI=http://duckiexp.local:11311
export ROS_IP=$(ipconfig getifaddr en0)
export VEHICLE_NAME=duckiexp

echo "=== Native ROS Noetic on Apple Silicon (VLM dashboard) ==="
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "ROS_IP: $ROS_IP"
echo "VLM server: $([ "$LOCAL" = true ] && echo 'disabled (--local, in-process VLM)' || echo "http://127.0.0.1:$PORT ($MODEL, $PRECISION)")"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ros_native
source ~/dev/native_ws/devel/setup.bash

VLM_SERVER_PID=""
VLM_SERVER_LOG="/tmp/vlm_server.log"

cleanup() {
    if [ -n "$VLM_SERVER_PID" ] && kill -0 "$VLM_SERVER_PID" 2>/dev/null; then
        echo "Stopping vlm_server.py (pid $VLM_SERVER_PID)..."
        kill "$VLM_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Sets the dashboard's offload-checkbox default state/URL (see vlm_dashboard_node.py) so
# --local doesn't also require unchecking it by hand.
export VLM_LOCAL="$LOCAL"
export VLM_SERVER_URL="http://127.0.0.1:$PORT"

if [ "$LOCAL" = false ]; then
    echo "Starting vlm_server.py in the background (log: $VLM_SERVER_LOG)..."
    python3 "$VLM_SERVER" --model "$MODEL" --precision "$PRECISION" --port "$PORT" > "$VLM_SERVER_LOG" 2>&1 &
    VLM_SERVER_PID=$!

    echo "Waiting for vlm_server.py to become healthy (loads the VLM, can take a while)..."
    for i in $(seq 1 120); do
        if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
            echo "vlm_server.py is up."
            break
        fi
        if ! kill -0 "$VLM_SERVER_PID" 2>/dev/null; then
            echo "vlm_server.py exited early, check $VLM_SERVER_LOG"
            exit 1
        fi
        sleep 2
    done
fi

rosrun cond_imitation_learning_pkg vlm_dashboard_node.py
