#ifndef HEAP_H
#define HEAP_H

#include <cuda_runtime.h>
#include<iostream>

#include"utils.h"
namespace tsdfvh {

/**
 * @brief      Class handling the indices of the voxel blocks.
 */

/** Index of the element of heap_ that contains the next available index */
//static __device__ volatile unsigned int heap_counter_;

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
            auto err=cudaMalloc(&heap_, sizeof(unsigned int) *( heap_size) );   
            cudaMalloc(&heap_counter_, sizeof(unsigned int) );   
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
            unsigned int idx = atomicSub( (int*)heap_counter_, 1);            
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
            unsigned int idx = atomicAdd( (int*)heap_counter_, 1);
            heap_[idx + 1] = ptr;
#endif
        }

        /** Vector of the indices currently assigned */
        unsigned int *heap_;

        __device__ void setCounter(uint n)
        {
                *heap_counter_=n;
        }

        __device__ uint getCounter() const
        {
                return *heap_counter_;
        }

        volatile uint *heap_counter_;
};

}  // namespace tsdfvh
#endif