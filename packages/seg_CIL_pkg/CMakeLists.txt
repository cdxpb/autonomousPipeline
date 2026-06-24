cmake_minimum_required(VERSION 3.0.2)
project(cond_imitation_learning_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/autonomous_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)