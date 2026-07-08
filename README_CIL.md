# ROS CIL Data Collector

A native ROS pipeline for Conditional Imitation Learning data collection in Duckietown. Uses a PyQt5 GUI via native X11 forwarding to drive the robot, tag intents, and synchronously record camera frames and wheel telemetry.

## Prerequisites
* Ubuntu Linux host (or VM with X11 support)
* Duckietown Shell (dts) installed
* Physical Duckiebot or Duckiematrix

## 1. Robot Setup

Physical Duckiebot:
Power on and connect to the same network. Note the hostname (e.g., duckiebot42).

Duckiematrix (Virtual):
Launch the matrix engine. The default vehicle map_0/vehicle_0 corresponds to the hostname golduck. 
Note: Ensure the simulation is Playing (press P) and Rendering (press R). If paused, camera data will not publish.

```
dts matrix engine run --sandbox --verbose
dts matrix run --browser --engine hostip
dts matrix attach golduck map_0/vehicle_0
```

## 2. Build and Run

Open your terminal in the project root.

1. Build the Docker image:  
   `dts devel build -f`

3. Run the container with native X11 forwarding. Replace <ROBOT_NAME> with your hostname (e.g., golduck):  
   dts devel run --robot duckiexp --cmd bash

## 3. Execute Nodes

Inside the container terminal run the following commands:
   ```
   source /code/catkin_ws/devel/setup.bash
   rosrun cond_imitation_learning_pkg autonomous_node.py --approach 5 --output_mode wheels
   ```

To hack into another shell:
docker exec -it dts-run-autonomouspipeline bash

To run ros stuff on that hacked shell: source /code/catkin_ws/devel/setup.bash

## 4. Native Mac Setup (Apple Silicon)

To run the pipeline natively on macOS without Docker, utilizing the Metal GPU (`mps`) for high-speed inference:

1. Install Conda / Miniforge.
2. Create the environment using the provided `native_env.yml`:
   ```bash
   conda env create -f native_env.yml
   ```
3. Setup the Native Workspace:
   ```bash
   mkdir -p ~/dev/native_ws/src
   cd ~/dev/native_ws/src
   git clone --depth 1 -b daffy https://github.com/duckietown/dt-ros-commons.git
   mv dt-ros-commons/packages/duckietown_msgs .
   rm -rf dt-ros-commons
   ln -s /path/to/your/autonomousPipeline/packages/cond_imitation_learning_pkg cond_imitation_learning_pkg
   ```
4. Build the Workspace:
   ```bash
   conda activate ros_native
   cd ~/dev/native_ws
   catkin build
   ```
5. Run the node natively using the provided script (make sure you are connected to the same Wi-Fi as the robot):
   ```bash
   bash ./run_native.sh
   ```