#ifndef KPARAMS_H
#define KPARAMS_H

#include <vector_types.h>
#include"cutil_math.h"
#include<vector>

typedef struct
{
    /** The resolution of the volume */
    uint3 volume_resolution = make_uint3(256,256,256);

    /** The starting position of the camrea.
     *  Usualy the center of the volume
     */
    float3 volume_direction = make_float3(4.0,4.0,4.0);

    /** The size of the volume in meters. */
    float3 volume_size = make_float3(8,8,8);


    /** Pyramids for ICP. */
    std::vector<int> pyramid = {10,5,4};

    /** mu parameter for integration */
    float mu = 0.1;

    /** ICP error threashold */
    float icp_threshold = 5.0e-01;    

    /** Resolution of the input images */
    uint2 inputSize;

    /** Camera matrix */
    float4 camera;

    ///Parameters for hashing
    /** Number of buckets */
    int num_buckets = 5000;
    /** Number of extra space on hash table
     * num_entries_=num_buckets*bucket_size;
    */
    int bucket_size = 2;

    /** Block size of hash entry assuming cubical
     * e.g. block_size*block_size*block_size
     */
    int block_size = 8;


    ///Parameters for covariance
    /** The noise of depth sensor. */
    float cov_z=0.02;

    /** A small covariance value. */
    float cov_small=1e-4;
    /** A big covariance value. */
    float cov_big=1e-2;

} kparams_t;


#endif // KPARAMS_H
