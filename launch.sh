#!/bin/bash
# Unified launcher for remote CIL components.
# Usage: ./launch.sh [command] [options...]
#
# Commands:
#   dashboard        Run plain CIL dashboard (telemetry + manual intent override).
#   dashboard_vlm    Run VLM-enabled dashboard (starts vlm_server.py in background).
#   autonomous       Run autonomous node with PyQt5 UI.
#   autonomous_vlm   Run VLM autonomous classifier node.
#   headless         Run headless autonomous node (for local dev).
#   dagger           Run DAgger data collection and UI.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/remote_env.sh"

COMMAND=$1
shift || true

if [ -z "$COMMAND" ]; then
    echo "Usage: $0 {dashboard|dashboard_vlm|autonomous|autonomous_vlm|headless|dagger} [options...]"
    exit 1
fi

case "$COMMAND" in
    dashboard)
        echo "=== Remote ROS Noetic ==="
        echo "ROS_MASTER_URI: $ROS_MASTER_URI"
        echo "ROS_IP: $ROS_IP"
        exec rosrun cond_imitation_learning_pkg remote_dashboard_node.py "$@"
        ;;

    autonomous)
        echo "=== Remote ROS Noetic ==="
        echo "ROS_MASTER_URI: $ROS_MASTER_URI"
        echo "ROS_IP: $ROS_IP"
        exec rosrun cond_imitation_learning_pkg autonomous_node.py --approach 7 "$@"
        ;;

    headless)
        echo "=== Remote ROS Noetic ==="
        echo "ROS_MASTER_URI: $ROS_MASTER_URI"
        echo "ROS_IP: $ROS_IP"
        exec rosrun cond_imitation_learning_pkg headless_autonomous_node.py --approach 7 "$@"
        ;;

    autonomous_vlm)
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        
        APPROACH=7
        while [[ "$#" -gt 0 ]]; do
            case $1 in
                --approach) APPROACH="$2"; shift ;;
            esac
            shift
        done
        
        echo "ROS_MASTER_URI: $ROS_MASTER_URI"
        echo "ROS_IP: $ROS_IP"
        echo "Approach: $APPROACH"
        python3 -c "import transformers" 2>/dev/null || echo "NOTE: 'transformers' not found in ros_native -- the VLM panel will fail to load until you 'pip install transformers' in this env (CIL/PilotNet still work without it)."
        exec rosrun cond_imitation_learning_pkg vlm_classifier_node.py --approach "$APPROACH"
        ;;

    dashboard_vlm)
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
            esac
            shift
        done
        
        VLM_SERVER="$SCRIPT_DIR/packages/cond_imitation_learning_pkg/src/vlm_server.py"
        echo "=== Remote ROS Noetic (VLM dashboard) ==="
        echo "ROS_MASTER_URI: $ROS_MASTER_URI"
        echo "ROS_IP: $ROS_IP"
        echo "VLM server: $([ "$LOCAL" = true ] && echo 'disabled (--local, in-process VLM)' || echo "http://127.0.0.1:$PORT ($MODEL, $PRECISION)")"
        
        VLM_SERVER_PID=""
        VLM_SERVER_LOG="/tmp/vlm_server.log"
        cleanup_vlm() {
            if [ -n "$VLM_SERVER_PID" ] && kill -0 "$VLM_SERVER_PID" 2>/dev/null; then
                echo "Stopping vlm_server.py (pid $VLM_SERVER_PID)..."
                kill "$VLM_SERVER_PID" 2>/dev/null || true
            fi
        }
        trap cleanup_vlm EXIT INT TERM
        
        export VLM_LOCAL="$LOCAL"
        export VLM_SERVER_URL="http://127.0.0.1:$PORT"
        
        if [ "$LOCAL" = false ]; then
            echo "Starting vlm_server.py in the background (log: $VLM_SERVER_LOG)..."
            python3 "$VLM_SERVER" --model "$MODEL" --precision "$PRECISION" --port "$PORT" > "$VLM_SERVER_LOG" 2>&1 &
            VLM_SERVER_PID=$!
            echo "Waiting for vlm_server.py to become healthy..."
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
        
        exec rosrun cond_imitation_learning_pkg vlm_dashboard_node.py
        ;;

    dagger)
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
            esac
            shift
        done
        
        if [ -z "$YOLO_BACKEND" ]; then YOLO_BACKEND="$BACKEND"; fi
        if [ -z "$PILOTNET_BACKEND" ]; then PILOTNET_BACKEND="$BACKEND"; fi
        
        echo "=== Remote DAgger collection ==="
        echo "Approach: $APPROACH | YOLO: $YOLO_BACKEND | PilotNet: $PILOTNET_BACKEND | Data dir: $DATA_DIR | fwd_speed: $FWD_SPEED | turn_bias: $TURN_BIAS"
        echo "ROS_MASTER_URI: $ROS_MASTER_URI"
        echo "ROS_IP: $ROS_IP"
        
        WS_DIR="$HOME/dev/native_ws"
        NEED_BUILD=0
        for pkg in dagger_pkg data_collector_pkg; do
            if [ ! -e "$WS_DIR/src/$pkg" ]; then
                ln -s "$SCRIPT_DIR/packages/$pkg" "$WS_DIR/src/$pkg"
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
        
        rosrun dagger_pkg dagger_node.py --approach "$APPROACH" --data-dir "$DATA_DIR" --fwd-speed "$FWD_SPEED" --turn-bias "$TURN_BIAS" --yolo-backend "$YOLO_BACKEND" --pilotnet-backend "$PILOTNET_BACKEND" &
        DAGGER_PID=$!
        trap "echo 'Stopping dagger_node...'; kill $DAGGER_PID 2>/dev/null" EXIT INT TERM
        rosrun data_collector_pkg ui_node.py
        ;;

    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: $0 {dashboard|dashboard_vlm|autonomous|autonomous_vlm|headless|dagger} [options...]"
        exit 1
        ;;
esac
