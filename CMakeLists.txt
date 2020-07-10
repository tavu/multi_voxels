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

catkin_package(CATKIN_DEPENDS 
    message_runtime 
    roscpp rospy std_msgs
)

include( ${CMAKE_CURRENT_LIST_DIR}/src/Algorithm/CMakeLists.txt)

#SET LD_LIBRARY_PATH ME TO PCL

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include 
    ${CMAKE_CURRENT_LIST_DIR}/src/Algorithm/src
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/src    
)

add_executable(multi_voxels_node 
#     src/kfusion_pose_graph_node.cpp
    src/multi_voxels_node.cpp
)

target_link_libraries(multi_voxels_node 
    multi_voxels_lib
    ${catkin_LIBRARIES}
)

