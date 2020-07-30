#ifndef KERNEL_GLOBALS_H
#define KERNEL_GLOBALS_H

#include "utils.h"
#include "volume.h"

/**
* @brief       Fuses srcVol into dstVol.
*
* @param[in]  dstVol Destination volume.
* @param[in]  srcVol Source volume.
* @param[in]  pose The inverse of the relative pose between volumes
* @param[in]  origin The origin of the volumes (usualy the center of the volume)
* @param[in]  maxweight The maximum weight of TSDF.
*/
__global__ void fuseVolumesKernel(Volume dstVol, 
                                  Volume srcVol, 
                                  const sMatrix4 pose,
                                  const float3 origin,
                                  const float maxweight);

/**
* @brief       Calculates depth from vertices
*
* @param[out] render The output depth.
* @param[in]  vertex The input vertices.
* @param[in]  K The camera matrix.
*/
__global__ void vertex2depthKernel(Image<float> render,
                                   Image<float3> vertex,
                                   const sMatrix4 K);
/**
* @brief        Renders an RGB image from volume
*
* @param[out]   render The output data.
* @param[in]    volume The input volume.
* @param[in]    vert The input vertices
* @param[in]    norm The input normals
*/
__global__ void renderRgbKernel(Image<uchar3> render,
                                const Volume volume,
                                Image<float3> vert,
                                Image<float3> norm);

/**
* @brief        Generates the gaussian kernel for the bilateral filter.
*
* @param[out]   out The output data.
* @param[in]    delta The delta parameter.
* @param[in]    radius The radius of gaussian.
*/
__global__ void generate_gaussian(Image<float> out,
                                  float delta,
                                  int radius);

/**
* @brief        Applies bilateral filter to depth.
*
* @param[out]   out The output data.
* @param[in]    in The input depth.
* @param[in]    e_d
* @param[in]    r The radius
*/
__global__ void bilateralFilterKernel(Image<float> out,
                                      const Image<float> in,
                                      const Image<float> gaussian,
                                      const float e_d,
                                      const int r);

/**
* @brief        Converts depth from millimeters to meters
*
* @param[out]   depth The output depth in meters.
* @param[in]    in The input depth in millimeters
*/
__global__ void mm2metersKernel( Image<float> depth, const Image<ushort> in );

/**
* @brief        Applies bilateral filter to depth together with half sampling.
*
* @param[out]   out The output data.
* @param[in]    in The input depth.
* @param[in]    e_d
* @param[in]    r The radius
*/
__global__ void halfSampleRobustImageKernel(Image<float> out,
                                            const Image<float> in,
                                            const float e_d,
                                            const int r);
/**
* @brief       Calculates vertices from depth data.
*
* @param[out] vertex The output vertices.
* @param[in]  depth The input depth.
* @param[in]  invK The inverse of camera matrix
*/
__global__ void depth2vertexKernel(Image<float3> vertex,
                                   const Image<float> depth,
                                   const sMatrix4 invK);

/**
* @brief       Calculates normals from vertices
*
* @param[out] normal The output normals.
* @param[in]  vertex The input vertices.
*/
__global__ void vertex2normalKernel(Image<float3> normal,const Image<float3> vertex);

/**
* @brief       Reduction for tracking
*/
__global__ void reduceKernel(float * out,
                             const Image<TrackData> J,
                             const uint2 size);

/**
* @brief      Tracking of ICP
*/
__global__ void trackKernel(Image<TrackData> output,
                            const Image<float3> inVertex,
                            const Image<float3> inNormal,
                            const Image<float3> refVertex,
                            const Image<float3> refNormal,
                            const sMatrix4 Ttrack,
                            const sMatrix4 view,
                            const float dist_threshold,
                            const float normal_threshold);

/**
* @brief       Raycast
*
* @param[out] pos3D The output vertices
* @param[out] normal The output normals
* @param[in]  volume The input volume
* @param[in]  view  The pose of raycast
* @param[in]  nearPlane  The nearPlane parameter.
* @param[in]  farPlane  The farPlane parameter.
* @param[in]  step  The small step of raycast
* @param[in]  largestep  The large step of raycast
* @param[in]  frame  The current frame sequence number
*/
__global__ void raycastKernel(Image<float3> pos3D,
                              Image<float3> normal,
                              const Volume volume,
                              const sMatrix4 view,
                              const float nearPlane,
                              const float farPlane,
                              const float step,
                              const float largestep,
                              int frame);

/**
* @brief       Integrates data into volume.
*
* @param[in]  vol The volume represents the map.
* @param[in]  depth The depth data
* @param[in]  rgb The rbg data
* @param[in]  invTrack The inverse of the integration pose
* @param[in]  K The camera matrix
* @param[in]  K The mu parameter
* @param[in]  maxweight The maximum weight of TSDF
*/
__global__ void integrateKernel(Volume vol,
                                const Image<float> depth,
                                const Image<uchar3> rgb,
                                const sMatrix4 invTrack,
                                const sMatrix4 K,
                                const float mu,
                                const float maxweight);

/**
* @brief       Renders track data for visualization.
*
* @param[out]  out The output data (RGB)
* @param[in]   data The input track data.
*/
__global__ void renderTrackKernel(Image<uchar3> out,
                                  const Image<TrackData> data);

/**
* @brief       Converts depth to RGB image for visualization
*
* @param[out]  out The output data (RGB)
* @param[in]   depth The input depth data.
* @param[in]   nearPlane The nearPlane parameter.
* @param[in]   farPlane The farPlane parameter.
*/
__global__ void renderDepthKernel(Image<uchar3> out,
                                  const Image<float> depth,
                                  const float nearPlane,
                                  const float farPlane);

/**
* @brief       Copies data from hash table to a continuous array.
*
* @param[in]   vol The volume.
* @param[in]   output The output data.
*/
__global__ void getVoxelData(Volume vol, short2 *output);

/**=================ICP COVARIANCE======================
 * According to:
   S. M. Prakhya, L. Bingbing, Y. Rui and W. Lin,
   "A closed-form estimate of 3D ICP covariance,"
  2015 14th IAPR International Conference on Machine Vision Applications (MVA),
  Tokyo, 2015, pp. 526-529, doi: 10.1109/MVA.2015.7153246.
*/

/**
* @brief       Calculates point to plane covariance
*              This is the first term (needs reduction)
*
* @param[in]   inVertex  The input vertices
* @param[in]   refVertex The model vertices
* @param[in]   refNormal The model normals
* @param[in]   trackData The track data for outliers
* @param[out]  outData   The output covariance
* @param[in]   Ttrack    The camera pose
* @param[in]   view      camMatrix*inverse(raycastPose);
* @param[in]   delta     Delta pose from ICP
* @param[in]   cov_big   A value representing a big covariance
*/
__global__ void icpCovarianceFirstTerm(const Image<float3> inVertex,
                                       const Image<float3> refVertex,
                                       const Image<float3> refNormal,
                                       const Image<TrackData> trackData,
                                       Image<sMatrix6> outData,
                                       const sMatrix4 Ttrack,
                                       const sMatrix4 view,
                                       const sMatrix4 delta,
                                       const float cov_big);

/**
* @brief       Calculates point to plane covariance
*              This is the first term (needs reduction)
*
* @param[in]   dataVertex  The input vertices
* @param[in]   modelVertex The model vertices
* @param[in]   modelNormals The model normals
* @param[in]   trackData The track data for outliers
* @param[out]  outData   The output covariance
* @param[in]   Ttrack    The camera pose
* @param[in]   view      camMatrix*inverse(raycastPose);
* @param[in]   delta     Delta pose from ICP
* @param[in]   cov_z     The noise of sensor
* @param[in]   cov_big   A value representing a big covariance
*/
__global__ void icpCovarianceSecondTerm(const Image<float3>  dataVertex,
                                         const Image<float3> modelVertex,
                                         const Image<float3> modelNormals,
                                         const Image<TrackData>  trackData,
                                         Image<sMatrix6> outData,
                                         const sMatrix4 Ttrack,
                                         const sMatrix4 view,
                                         const sMatrix4 delta,
                                         float cov_z,
                                         const float cov_big);

/**
* @brief       Calculates point to point covariance
*              This is the first term (needs reduction)
*
* @param[in]   vert  The input vertices
* @param[in]   vertSize  The size of  vert array.
* @param[in]   prevVert  The previous vertices
* @param[in]   prevVertSize The size of prevVert array.
* @param[in]   sourceCorr The indexes of the correspondences of vert array
* @param[in]   targetCorr The indexes of the correspondences of prevVert array
* @param[in]   correspSize The sizeo of sourceCorr and targetCorr arrays
* @param[in]   delta     Delta pose from ICP
* @param[out]  outData   The output data
* @param[in]   cov_big   A value representing a big covariance
*/
__global__ void point2PointCovFirstTerm(const float3 *vert,
                                        int vertSize,
                                        const float3 *prevVert,
                                        int prevVertSize,
                                        const int *sourceCorr,
                                        const int *targetCorr,
                                        int correspSize,
                                        sMatrix4 delta,
                                        sMatrix6 *outData,
                                        const float cov_big);

/**
* @brief       Calculates point to point covariance
*              This is the second term (needs reduction)
*
* @param[in]   vert  The input vertices
* @param[in]   vertSize  The size of  vert array.
* @param[in]   prevVert  The previous vertices
* @param[in]   prevVertSize The size of prevVert array.
* @param[in]   sourceCorr The indexes of the correspondences of vert array
* @param[in]   targetCorr The indexes of the correspondences of prevVert array
* @param[in]   correspSize The sizeo of sourceCorr and targetCorr arrays
* @param[in]   delta     Delta pose from ICP
* @param[in]   cov_z The noise of the sensor
* @param[out]  outData   The output data
* @param[in]   cov_big   A value representing a big covariance
*/
__global__ void point2PointCovSecondTerm(const float3 *vert,
                                        int vertSize,
                                        const float3 *prevVert,
                                        int prevVertSize,
                                        const int *sourceCorr,
                                        const int *targetCorr,
                                        int correspSize,
                                        const sMatrix4 delta,
                                        float cov_z,
                                        sMatrix6 *outData,
                                        const float cov_big);
#endif // KERNEL_GLOBALS_H
