// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tsdfvh/hash_table.h"
#include <iostream>

#define THREADS_PER_BLOCK 512

namespace tsdfvh
{
__global__ void InitEntriesKernel(HashEntry *entries, int num_entries);

__global__ void InitHeapKernel(Heap *heap,
                               VoxelBlock *voxel_blocks,
                               int num_blocks,
                               int block_size);


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
    cudaMallocManaged(&entries_, sizeof(HashEntry) * num_entries_);
    cudaMallocManaged(&voxels_, sizeof(Voxel) * block_size * block_size *
                                    block_size * num_blocks);
    cudaMallocManaged(&voxel_blocks_,
                    sizeof(VoxelBlock) * num_blocks);
    cudaMallocManaged(&heap_, sizeof(Heap));
    cudaDeviceSynchronize();
    for (size_t i = 0; i < num_blocks; i++)
    {
        voxel_blocks_[i].Init(&(voxels_[i * block_size * block_size * block_size]),
                              block_size);
    }

    heap_->Init(num_blocks);
    int threads_per_block = THREADS_PER_BLOCK;
    int thread_blocks = (num_entries_ + threads_per_block - 1) / threads_per_block;

    InitEntriesKernel<<<thread_blocks, threads_per_block>>>(entries_,
                                                            num_entries_);
    cudaDeviceSynchronize();

    thread_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
    InitHeapKernel<<<thread_blocks, threads_per_block>>>(heap_, voxel_blocks_,
                                                        num_blocks, block_size);
    cudaDeviceSynchronize();
}

void HashTable::Free()
{
  cudaFree(entries_);
  cudaFree(voxels_);
  cudaFree(voxel_blocks_);
  cudaFree(heap_);
}

__global__ void InitEntriesKernel(HashEntry *entries, int num_entries) 
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < num_entries; i += stride) 
  {
    entries[i].pointer = kFreeEntry;
    entries[i].position = make_int3(0, 0, 0);
  }
}

__global__ void InitHeapKernel(Heap *heap, 
                               VoxelBlock *voxel_blocks,
                               int num_blocks, 
                               int block_size) 
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

    for (int j = 0; j < block_size * block_size * block_size; j++) 
    {
      voxel_blocks[i].at(j).sdf = 0;
      voxel_blocks[i].at(j).color = make_float3(0, 0, 0);
      voxel_blocks[i].at(j).weight = 0;
    }
  }
}

}  // namespace tsdfvh
