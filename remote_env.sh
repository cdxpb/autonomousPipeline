#!/bin/bash
# Common environment setup for remote execution (Linux/macOS)

export ROS_MASTER_URI=http://duckiexp.local:11311
export VEHICLE_NAME=duckiexp

if command -v ipconfig >/dev/null 2>&1; then
    export ROS_IP=$(ipconfig getifaddr en0)
else
    export ROS_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
    if [ -z "$ROS_IP" ]; then
        export ROS_IP=127.0.0.1
    fi
fi

if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate ros_native
fi

if [ -f "$HOME/dev/native_ws/devel/setup.bash" ]; then
    source "$HOME/dev/native_ws/devel/setup.bash"
fi
