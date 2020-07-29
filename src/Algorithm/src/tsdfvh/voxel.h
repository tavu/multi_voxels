#ifndef VOXEL_H
#define VOXEL_H

#include <cuda_runtime.h>
#include"constant_parameters.h"
namespace tsdfvh 
{

typedef unsigned char uchar;

class Voxel
{
    public:
        short sdf;
        float3 color;
        short weight;

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
            return color;
        }

        __device__ inline
        void setColor(const float3 &col)
        {
            color=col;
        }
};

}  // namespace tsdfvh

typedef tsdfvh::Voxel voxel_t;

#endif
