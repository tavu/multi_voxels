cmake_minimum_required(VERSION 3.3)
project(multi_voxels_lib)

find_package(CUDA)

add_compile_options(-std=c++11)
# link_directories("${CMAKE_CURRENT_LIST_DIR}/dependencies/isam/lib/")

# set(TOON_INCLUDE_PATHS "${CMAKE_CURRENT_LIST_DIR}/dependencies/TooN")
# set(EIGEN_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/dependencies/eigen")
# set(ISAM_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/dependencies/isam/include")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(TOON_INCLUDE_PATHS "$ENV{HOME}/src/TooN")
set(EIGEN_INCLUDE_DIRS "/usr/local/include/eigen3")

# set(eigen3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS} )
# find_package( OpenCV REQUIRED )

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TOON_INCLUDE_PATHS}
    ${EIGEN_INCLUDE_DIRS}
#     ${SUITESPARSE_INCLUDE_DIR}
#     ${OpenCV_INCLUDE_DIRS}
#     ${CMAKE_CURRENT_LIST_DIR}/src/teaser
#     ${CMAKE_CURRENT_LIST_DIR}/src/pmc
#     ${ISAM_INCLUDE_DIRS}
)

set(cuda_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/kernels.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/kfusion.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/kernelscalls.cu
)

set(cpp_srcs
    ${CMAKE_CURRENT_LIST_DIR}/src/volume.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/fusionHandler.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/src/featuredetector.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/src/keyptsmap.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/src/Isam.cpp     
#    ${CMAKE_CURRENT_LIST_DIR}/src/saveData.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/src/SiftCovEstimator.cpp
)


SET( CUDA_PROPAGATE_HOST_FLAGS OFF )

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC;" )
cuda_add_library(multi_voxels_cuda_lib   STATIC  ${cuda_srcs}  OPTIONS -std=c++11; -Xcompiler; -fPIC; 
    -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
    -I${CMAKE_CURRENT_LIST_DIR}/src/
    -I${TOON_INCLUDE_PATHS}
    -I${EIGEN3_INCLUDE_DIR}
)
SET_TARGET_PROPERTIES(multi_voxels_cuda_lib PROPERTIES COMPILE_FLAGS "-fPIC" )

add_library(multi_voxels_lib 
    ${cpp_srcs}
#     ${sift_src}
#     ${teaser_src}
#     ${pmc_src}
)

# message(WARNING ${OPEN3D_DEF})

# if(${USE_OPEN3D})
# #     SET_TARGET_PROPERTIES(multi_voxels_lib PROPERTIES COMPILE_FLAGS "-fPIC -fopenmp -DUSE_OPEN3D -ggdb" )
# else()
SET_TARGET_PROPERTIES(multi_voxels_lib PROPERTIES COMPILE_FLAGS "-fPIC" )
# endif()
    
# find_package(OpenMP)
# set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#Comment ISAM if use G2O
target_link_libraries(multi_voxels_lib
#     ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}    
    multi_voxels_cuda_lib
#     isam
#     cholmod
)
