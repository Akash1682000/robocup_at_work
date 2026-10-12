cmake_minimum_required(VERSION 3.8)
project(robocup_grasp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)

link_directories(${gazebo_dev_LIBRARY_DIRS})

add_library(robocup_grasp_plugin SHARED src/grasp_plugin.cpp)
target_include_directories(robocup_grasp_plugin PUBLIC ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(robocup_grasp_plugin ${GAZEBO_LIBRARIES})
ament_target_dependencies(robocup_grasp_plugin
  gazebo_dev
  gazebo_ros
  rclcpp
  std_srvs
)

install(TARGETS robocup_grasp_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.dsv.in")
ament_package()
