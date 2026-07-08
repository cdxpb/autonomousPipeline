#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

APPROACH="${APPROACH:-7}"
BACKEND="${BACKEND:-onnx}"

echo "======================================================="
echo " Duckiebot Headless Inference Launcher"
echo " Approach : $APPROACH"
echo " Backend  : $BACKEND"
echo "======================================================="

# Re-build catkin workspace so rosrun can find newly synced scripts
source /opt/ros/noetic/setup.sh
catkin build --workspace /code/catkin_ws/ --no-status -q 2>&1 | tail -5
source /code/catkin_ws/devel/setup.bash

# Verify models are present
MODEL_BASE="/data/models"
if [ ! -d "$MODEL_BASE/pilotnet" ]; then
    echo ""
    echo "ERROR: Models not found at $MODEL_BASE/pilotnet/"
    echo "Transfer them from your Mac first:"
    echo "  ./transfer_models.sh --all"
    exit 1
fi

echo "Models OK at $MODEL_BASE"
echo "Starting headless_autonomous_node (approach=$APPROACH, backend=$BACKEND)..."

dt-exec rosrun cond_imitation_learning_pkg headless_autonomous_node.py \
    --approach "$APPROACH" \
    _backend:="$BACKEND"

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
