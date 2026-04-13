cmake_minimum_required(VERSION 3.8)
project(active_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseEstimateSample.msg"
  "srv/EvaluatePoseConfidence.srv"
  "srv/PlanNBV.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  active_perception/cylinder_finder.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cylinder_finder
)

install(PROGRAMS
  active_perception/pose_estimator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pose_estimator
)

install(PROGRAMS
  active_perception/box_finder.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME box_finder
)

install(PROGRAMS
  active_perception/confidence_evaluator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME confidence_evaluator
)

install(PROGRAMS
  active_perception/nbv_planner.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME nbv_planner
)

install(PROGRAMS
  active_perception/orchestrator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME orchestrator
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
