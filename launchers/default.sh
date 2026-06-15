#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

echo "VNC Environment is running."
echo "Container is now awake and waiting for UI interactions."

# Use dt-exec to run a blocking command. This keeps the launchfile 
# from joining, keeping your container alive indefinitely.
dt-exec sleep infinity

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join