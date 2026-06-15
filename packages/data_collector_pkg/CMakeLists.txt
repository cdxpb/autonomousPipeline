cmake_minimum_required(VERSION 3.0.2)
project(data_collector_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  duckietown_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  src/ui_node.py
  src/logger_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)