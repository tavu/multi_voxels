cmake_minimum_required(VERSION 3.3)
project(multi_voxels)

find_package(CUDA)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    rospy 
    std_msgs
    image_transport
    tf2
    tf2_ros
    cv_bridge
)

find_package(CUDA)

set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(TOON_INCLUDE_PATHS "$ENV{HOME}/src/TooN")
# set(EIGEN_INCLUDE_DIRS "/usr/local/include/eigen3")

catkin_package(CATKIN_DEPENDS 
    message_runtime 
    roscpp 
    rospy 
    std_msgs 
    g2o_slam
    message_generation
)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TOON_INCLUDE_PATHS}
    ${CMAKE_CURRENT_LIST_DIR}/include 
    ${CMAKE_CURRENT_LIST_DIR}/src/Algorithm/include
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/src    
    ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)

# include( ${CMAKE_CURRENT_LIST_DIR}/src/Algorithm/CMakeLists.txt)
add_subdirectory( ${CMAKE_CURRENT_LIST_DIR}/src/Algorithm/)

add_executable(multi_voxels_node 
    src/multi_voxels_node.cpp
)

target_link_libraries(multi_voxels_node 
    multi_voxels_lib
    ${catkin_LIBRARIES}
)

add_dependencies(multi_voxels_node ${catkin_EXPORTED_TARGETS})