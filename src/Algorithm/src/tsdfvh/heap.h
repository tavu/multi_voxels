#pragma once

#include <cuda_runtime.h>

namespace tsdfvh {

/**
 * @brief      Class handling the indices of the voxel blocks.
 */
class Heap
{
    public:
        /**
        * @brief      Allocates the memory necessary for the heap
        *
        * @param[in]  heap_size  The maximum number of indices that can be assigned
        */
        inline void Init(int heap_size)
        {
            cudaMallocManaged(&heap_, sizeof(unsigned int) * heap_size);
        }

        /**
        * @brief      Function to request an index to be assigned to a voxel block.
        *
        * @return     The index to be assigned to a voxel block (i.e., consumed from
        *             the heap).
        */
        __device__ inline unsigned int Consume()
        {
#ifdef __CUDACC__
            unsigned int idx = atomicSub(&heap_counter_, 1);
            return heap_[idx];
#endif
        }

        /**
        * @brief      Frees the given index.
        *
        * @param[in]  ptr   The index to be freed (i.e., appended to the heap)
        */
        __device__ inline void Append(unsigned int ptr)
        {
#ifdef __CUDACC__
            unsigned int idx = atomicAdd(&heap_counter_, 1);
            heap_[idx + 1] = ptr;
#endif
        }

        /** Vector of the indices currently assigned */
        unsigned int *heap_;

        /** Index of the element of heap_ that contains the next available index */
        unsigned int heap_counter_;
};

}  // namespace tsdfvh
