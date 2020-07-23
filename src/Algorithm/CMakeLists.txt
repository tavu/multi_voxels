cmake_minimum_required(VERSION 3.3)
project(multi_voxels_lib)

find_package(CUDA)

add_compile_options(-std=c++11)

set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(TOON_INCLUDE_PATHS "$ENV{HOME}/src/TooN")
set(EIGEN_INCLUDE_DIRS "/usr/local/include/eigen3")

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TOON_INCLUDE_PATHS}
    ${EIGEN_INCLUDE_DIRS}
)

set(cuda_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/kernels.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/kfusion.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/kernelscalls.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/tsdfvh/hash_table.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/tsdfvh/heap.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/volume.cu
)

# set(cpp_srcs
# #     ${CMAKE_CURRENT_LIST_DIR}/src/volume.cpp
# #     ${CMAKE_CURRENT_LIST_DIR}/src/fusionHandler.cpp
# )


SET( CUDA_PROPAGATE_HOST_FLAGS OFF )

find_package(CUDA REQUIRED)
set( CUDA_NVCC_FLAGS 
    "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC;-arch=sm_50;" 
)

cuda_add_library(multi_voxels_lib   STATIC  ${cuda_srcs}  
    OPTIONS -std=c++11; -Xcompiler; -fPIC;
    -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
    -I${CMAKE_CURRENT_LIST_DIR}/src/
    -I${TOON_INCLUDE_PATHS}
    -I${EIGEN3_INCLUDE_DIR}
    -I${CMAKE_CURRENT_LIST_DIR}/src/tsdfvh/
)
SET_TARGET_PROPERTIES(multi_voxels_lib PROPERTIES COMPILE_FLAGS "-fPIC" )

# add_library(multi_voxels_lib 
#     ${cpp_srcs}
# )
# 
# SET_TARGET_PROPERTIES(multi_voxels_lib PROPERTIES COMPILE_FLAGS "-fPIC" )
# 
# target_link_libraries(multi_voxels_lib
#     ${CUDA_LIBRARIES}    
#     multi_voxels_cuda_lib
# )
