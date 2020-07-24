// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tsdfvh/hash_table.h"
#include <iostream>

#define THREADS_PER_BLOCK 512

namespace tsdfvh
{
__global__ void initEntriesKernel(HashEntry *entries, int num_entries);

__global__ void initHeapKernel(Heap *heap,
                               int num_blocks);
__global__ void initVoxelsKernel(Voxel *voxels, int size );


void HashTable::Init(int num_buckets,
                     int bucket_size,
                     int num_blocks,
                     int block_size)
{
    num_buckets_ = num_buckets;
    bucket_size_ = bucket_size;
    num_entries_ = num_buckets * bucket_size;
    num_blocks_ = num_blocks;
    block_size_ = block_size;
    num_allocated_blocks_ = 0;

    int vsize=block_size*block_size*block_size*num_blocks;

    cudaMalloc( &entries_, sizeof(HashEntry)*num_entries_);
    cudaMalloc( &voxels_, sizeof(Voxel)*vsize);
    //TODO how to allocate size on device and store it on device pointer?
    cudaMallocManaged(&heap_, sizeof(Heap));

    heap_->Init(num_blocks);
}

void HashTable::setEmpty()
{
    int threads_per_block = THREADS_PER_BLOCK;
    int thread_blocks = (num_entries_ + threads_per_block - 1) / threads_per_block;

    int vsize=block_size_*block_size_*block_size_*num_blocks_;

    initEntriesKernel<<<thread_blocks, threads_per_block>>>(entries_,num_entries_);

    thread_blocks = (num_blocks_ + threads_per_block - 1) / threads_per_block;
    initHeapKernel<<<thread_blocks, threads_per_block>>>(heap_, num_blocks_);

    initVoxelsKernel<<<vsize/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(voxels_,vsize);
    cudaDeviceSynchronize();
}

void HashTable::Free()
{
    cudaFree(entries_);
    cudaFree(voxels_);
    cudaFree(heap_);
}

__global__ void initEntriesKernel(HashEntry *entries, int num_entries)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < num_entries; i += stride)
    {
        entries[i].pointer = kFreeEntry;
        entries[i].position = make_int3(0, 0, 0);
    }
}

__global__ void initHeapKernel(Heap *heap,int num_blocks)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (index == 0)
    {
        heap->heap_counter_ = num_blocks - 1;
    }

    for (int i = index; i < num_blocks; i += stride)
    {
        heap->heap_[i] = num_blocks - i - 1;
    }
}

__global__ void initVoxelsKernel(Voxel *voxels, int size )
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size)
    {
        voxels[idx].color=make_float3(0.0,0.0,0.0);
        voxels[idx].weight=0.0;
        voxels[idx].sdf=1.0;
    }
}

}  // namespace tsdfvh
