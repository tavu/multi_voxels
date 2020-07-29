#ifndef VOXEL_H
#define VOXEL_H

#include <cuda_runtime.h>
#include"constant_parameters.h"

#include<cuda_fp16.h>

namespace tsdfvh 
{

typedef unsigned char uchar;

class Voxel
{
    public:
        short sdf;
        short weight;
        short3 color;
#ifdef __CUDACC__
        __half r;
        __half g;
        __half b;

        __device__ inline
        float getWeight() const
        {
            return static_cast<float>(weight);
        }

        __device__ inline
        void setWeight(float w)
        {
            weight=static_cast<short>(w+0.5);
        }

        __device__ inline
        float getTsdf() const
        {
            return static_cast<float>(sdf)*short2float;
        }

        __device__ inline
        void setTsdf(float d)
        {
            sdf=static_cast<short>(d*float2short);
        }

        __device__ inline
        float3 getColor() const
        {
            return make_float3(__half2float(r),
                               __half2float(g),
                               __half2float(b));
        }

        __device__ inline
        void setColor(const float3 &col)
        {
            r=__float2half(col.x);
            g=__float2half(col.y);
            b=__float2half(col.z);
        }
#endif
};

}  // namespace tsdfvh

typedef tsdfvh::Voxel voxel_t;

#endif
