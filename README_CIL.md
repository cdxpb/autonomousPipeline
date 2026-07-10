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

## 5. Native Jetson Setup (on the Duckiebot itself, no Docker)

Confirmed on `duckiexp`: Ubuntu 18.04.6 (Bionic), Python 3.6.9, aarch64, `nvidia-l4t-core 32.7.6`,
4GB RAM. Runtime backend is TensorRT + PyCUDA, not raw PyTorch (torch's CUDA kernel loading
alone can spike RAM/swap by ~1.8GB on a 4GB Nano).

1. From your Mac, sync source + models to the robot:
   ```bash
   ./sync_repo_to_bot.sh
   ./transfer_models.sh --all
   ```
2. SSH into the robot and run each setup stage, reviewing output between stages:
   ```bash
   ssh duckie@duckiexp.local
   cd ~/dev/autonomousPipeline
   bash setup_native_jetson.sh preflight        # read-only: checks CUDA/TensorRT/ROS state
   bash setup_native_jetson.sh install-ros       # apt install ros-melodic-ros-base
   bash setup_native_jetson.sh install-pycuda    # pycuda + verify tensorrt python bindings
   bash setup_native_jetson.sh build-workspace   # catkin workspace, symlinks cond_imitation_learning_pkg
   ```
3. Build TensorRT engines from the existing ONNX models (still on the robot):
   ```bash
   python3 packages/cond_imitation_learning_pkg/src/export_trt.py --all
   ```
4. Run natively:
   ```bash
   bash run_robot_native.sh --approach 7
   ```

`preflight` checks whether the Docker container running `roscore` is on `NetworkMode=host` --
required for the native process to reach the ROS master on `localhost`.

If RAM is tight, `./bot_containers.sh` lists running containers with live memory usage so you can
stop ones you don't need. 