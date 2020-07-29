#include "tsdfvh/hash_table.h"
#include <iostream>
#include "utils.h"
#define THREADS_PER_BLOCK 512

namespace tsdfvh
{
__global__ void initEntriesKernel(volatile HashEntry *entries, int num_entries);

__global__ void initHeapKernel(Heap heap,
                               int num_blocks, int num_buckets);
__global__ void initVoxelsKernel(voxel_t *voxels, int size );


void HashTable::Init(int num_buckets,
                     int bucket_size,
                     int block_size)
{
    block_size_ = block_size;
    num_buckets_ = num_buckets;
    bucket_size_ = bucket_size;
    num_entries_=num_buckets*bucket_size;

    int vsize=block_size*block_size*block_size*num_entries_;


    heap_size_=num_entries_-num_buckets;

    cudaMalloc( &entries_, sizeof(HashEntry)*num_entries_);
    cudaMalloc( &voxels_, sizeof(Voxel)*vsize);
    heap_.Init(heap_size_);
}

void HashTable::setEmpty()
{
    int threads_per_block = THREADS_PER_BLOCK;
    int thread_blocks = (num_entries_ + threads_per_block ) / threads_per_block;

    int vsize=block_size_*block_size_*block_size_*num_entries_;
    initEntriesKernel<<<(num_entries_+THREADS_PER_BLOCK)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(entries_,num_entries_);
    printCUDAError();

    thread_blocks = (heap_size_ + threads_per_block ) / threads_per_block;
    initHeapKernel<<<thread_blocks, threads_per_block>>>(heap_, heap_size_,num_buckets_);
    printCUDAError();

    initVoxelsKernel<<<(vsize+THREADS_PER_BLOCK)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(voxels_,vsize);
    printCUDAError();
}

void HashTable::Free()
{
    cudaFree( (void*)entries_);
    cudaFree( (void*)voxels_);
    //cudaFree(heap_);
}

__global__ void initEntriesKernel(volatile HashEntry *entries, int num_entries)
{
     int idx = blockIdx.x*blockDim.x + threadIdx.x;
     if(idx<num_entries)
     {
         entries[idx].next_ptr = kFreeEntry;
         entries[idx].position.x = 0;
         entries[idx].position.y = 0;
         entries[idx].position.z = 0;
     }
}

__global__ void initHeapKernel(Heap heap,int heap_size_, int num_buckets)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
    {
        heap.setCounter(heap_size_ - 1);        
    }

    if(index<heap_size_)
    {
        heap.heap_[index] = index + num_buckets ;
    }
}

__global__ void initVoxelsKernel(voxel_t *voxels, int size )
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size)
    {
        voxels[idx].color.x=0.0;
        voxels[idx].color.y=0.0;
        voxels[idx].color.z=0.0;

        voxels[idx].setWeight(0.0);
        voxels[idx].setTsdf(1.0);
    }
}

}  // namespace tsdfvh
