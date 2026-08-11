# Autonomous Pipeline

<div align="center">
  <video src="website/assets/vlmCompiledCompressed.mp4" width="800" controls autoplay loop muted></video>
  <br>
  <em>Given text instruction as a high level plan, we parse a queue of maneuvers. Based on the queue, at each frame we prompt our VLM and get back an intent. Based on the predicted intent and the segmentation mask of our current frame, the Conditional Imitation Learning controller predicts wheel velocities.</em>
</div>

<br>

ROS pipeline for Conditional Imitation Learning (CIL) and Vision-Language Model (VLM) navigation on Duckietown.

This repository includes data collection, a PyQt5 remote dashboard, and autonomous driving using PilotNet and YOLO semantic segmentation. Models are trained in PyTorch, exported to ONNX, and run via TensorRT on the Jetson Nano.

## Setup

The system is split between the robot (Duckiebot) and your remote machine.

### 1. Robot Setup
Turn on your robot and connect to its network. Note the hostname (e.g. `duckiebot42`).

If you are using the simulator (Duckiematrix), launch the matrix engine and attach to the vehicle:
```bash
dts matrix engine run --sandbox --verbose
dts matrix run --browser --engine hostip
dts matrix attach golduck map_0/vehicle_0
```

### 2. Remote Machine Setup (macOS / Linux)
You need Conda installed.

```bash
conda env create -f native_env.yml
conda activate ros_native

mkdir -p ~/dev/native_ws/src
cd ~/dev/native_ws/src
git clone --depth 1 -b daffy https://github.com/duckietown/dt-ros-commons.git
mv dt-ros-commons/packages/duckietown_msgs .
rm -rf dt-ros-commons

# Symlink the packages into your workspace
ln -s /path/to/autonomousPipeline/packages/cond_imitation_learning_pkg cond_imitation_learning_pkg
ln -s /path/to/autonomousPipeline/packages/data_collector_pkg data_collector_pkg

cd ~/dev/native_ws
catkin build
```

## Usage

### Data Collection
To collect behavioral cloning data, run the logger and the UI. The data will be saved locally to `~/Desktop/my_dataset/cil_dataset_YYYYMMDD-HHMMSS/`.

Run these in two separate terminal tabs:
```bash
rosrun data_collector_pkg logger_node.py
rosrun data_collector_pkg ui_node.py
```

**Controls** (make sure the UI window is focused):
* W/A/S/D: Drive
* R: Toggle data logging on/off
* I/J/L/K: Intents (Straight, Left, Right, Lane Following)

### Autonomous Navigation
To run the autonomous driving pipeline directly from your remote machine (uses PyTorch by default):
```bash
./launch.sh autonomous
```

For Vision-Language Model (VLM) guided navigation:
```bash
./launch.sh dashboard_vlm --port 8000
```

To export PyTorch models to ONNX:
```bash
python3 packages/cond_imitation_learning_pkg/src/export_model.py --format onnx --approach 7
```

## Jetson Deployment (TensorRT)

To run inference entirely on the Duckiebot's Jetson Nano:

1. Sync the repo and models to the robot:
   ```bash
   ./scripts/sync_repo_to_bot.sh
   ./scripts/transfer_models.sh --all
   ```
2. SSH into the robot and build the workspace:
   ```bash
   ./scripts/setup_native_jetson.sh
   ```
3. Build TensorRT engines from the ONNX models:
   ```bash
   python3 packages/cond_imitation_learning_pkg/src/export_model.py --format trt --approach 7
   ```
4. Run natively on the robot:
   ```bash
   ./launchers/run_robot_native.sh --approach 7
   ```