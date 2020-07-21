#ifndef VOXEL_H
#define VOXEL_H

#include <cuda_runtime.h>

namespace tsdfvh 
{

typedef unsigned char uchar;

struct Voxel 
{
    float sdf;
    float3 color;
    float weight;

    __host__ __device__ void Combine(const Voxel& voxel, uchar max_weight) 
    {
        color.x = static_cast<uchar>(
            (static_cast<float>(color.x) * static_cast<float>(weight) +
                static_cast<float>(voxel.color.x) * static_cast<float>(voxel.weight)) /
                (static_cast<float>(weight) +
            static_cast<float>(voxel.weight)));
        color.y = static_cast<uchar>(
            (static_cast<float>(color.y) * static_cast<float>(weight) +
                static_cast<float>(voxel.color.y) * static_cast<float>(voxel.weight)) /
                (static_cast<float>(weight) +
            static_cast<float>(voxel.weight)));
        color.z = static_cast<uchar>(
            (static_cast<float>(color.z) * static_cast<float>(weight) +
                static_cast<float>(voxel.color.z) * static_cast<float>(voxel.weight)) /
                (static_cast<float>(weight) +
            static_cast<float>(voxel.weight)));

        sdf = (sdf * static_cast<float>(weight) +
                voxel.sdf * static_cast<float>(voxel.weight)) /
                    (static_cast<float>(weight) + static_cast<float>(voxel.weight));

        weight = weight + voxel.weight;
        if (weight > max_weight) 
            weight = max_weight;
    }
};

}  // namespace tsdfvh

#endif
