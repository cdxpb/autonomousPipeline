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
   `dts devel run -X --robot <ROBOT_NAME> --cmd bash`

## 3. Execute Nodes

Inside the container terminal run the following commands:
   ```
   source /code/catkin_ws/devel/setup.bash
   rosrun cond_imitation_learning_pkg autonomous_node.py
   ```

To hack into another shell:
docker exec -it dts-run-autonomouspipeline bash

To run ros stuff on that hacked shell: source /code/catkin_ws/devel/setup.bash