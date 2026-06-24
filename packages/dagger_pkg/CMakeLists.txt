cmake_minimum_required(VERSION 3.0.2)
project(dagger_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  duckietown_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  src/dagger_node.py
  src/dagger_trainer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)