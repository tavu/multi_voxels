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
#     g2o_slam
)

catkin_package(CATKIN_DEPENDS 
    message_runtime 
    roscpp 
    rospy 
    std_msgs 
    g2o_slam
    message_generation
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include 
    ${CMAKE_CURRENT_LIST_DIR}/src/Algorithm/src
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/src    
    ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
    ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)

include( ${CMAKE_CURRENT_LIST_DIR}/src/Algorithm/CMakeLists.txt)

add_executable(multi_voxels_node 
    src/multi_voxels_node.cpp
)

target_link_libraries(multi_voxels_node 
    multi_voxels_lib
    ${catkin_LIBRARIES}
)

add_dependencies(multi_voxels_node ${catkin_EXPORTED_TARGETS})