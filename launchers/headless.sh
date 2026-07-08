#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

APPROACH="${APPROACH:-7}"
BACKEND="${BACKEND:-pytorch}"

echo "======================================================="
echo " Duckiebot Headless Inference Launcher"
echo " Approach : $APPROACH"
echo " Backend  : $BACKEND"
echo "======================================================="

# Re-build catkin workspace so rosrun can find newly synced scripts
# Force-remove the stale devel wrapper so catkin always reinstalls it fresh
rm -f /code/catkin_ws/devel/lib/cond_imitation_learning_pkg/headless_autonomous_node.py
source /opt/ros/noetic/setup.sh
# Set the global camera framerate down to 15 FPS to reduce source load
rosparam set /$VEHICLE_NAME/camera_node/framerate 15

catkin build --workspace /code/catkin_ws/ --no-status 2>&1 | tail -3
source /code/catkin_ws/devel/setup.bash

echo "=== CHECKING ACTUAL SCRIPT CONTENT ==="
head -n 80 /code/catkin_ws/src/autonomousPipeline/packages/cond_imitation_learning_pkg/src/headless_autonomous_node.py | tail -n 15
echo "======================================"

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
