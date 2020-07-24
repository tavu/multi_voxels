#include "tsdfvh/hash_table.h"
#include <iostream>
#include "utils.h"
#define THREADS_PER_BLOCK 512

namespace tsdfvh
{
__global__ void initEntriesKernel(HashEntry *entries, int num_entries);

__global__ void initHeapKernel(Heap *heap,
                               int num_blocks, int num_buckets);
__global__ void initVoxelsKernel(Voxel *voxels, int size );


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

    printf("block_size:%d\n",block_size);
    printf("num_entries_:%d\n",num_entries_);
    printf("heap_size_:%d\n",heap_size_);
    printf("vsize:%d\n",vsize);

    cudaMalloc( &entries_, sizeof(HashEntry)*num_entries_);
    cudaMalloc( &voxels_, sizeof(Voxel)*vsize);

    cudaMallocManaged(&heap_, sizeof(Heap));
    heap_->Init(heap_size_);




//    num_buckets_ = num_buckets;
//    bucket_size_ = bucket_size;
//    num_entries_ = num_buckets * bucket_size;
//    num_blocks_ = num_blocks;
//    block_size_ = block_size;
//    num_allocated_blocks_ = 0;

//    int vsize=block_size*block_size*block_size*num_blocks;

//    cudaMalloc( &entries_, sizeof(HashEntry)*num_entries_);
//    cudaMalloc( &voxels_, sizeof(Voxel)*vsize);
//    //TODO how to allocate size on device and store it on device pointer?
//    cudaMallocManaged(&heap_, sizeof(Heap));

//    heap_->Init(num_blocks);
}

void HashTable::setEmpty()
{
    int threads_per_block = THREADS_PER_BLOCK;
    int thread_blocks = (num_entries_ + threads_per_block - 1) / threads_per_block;

    int vsize=block_size_*block_size_*block_size_*num_entries_;

    printf("initEntriesKernel\n");
    initEntriesKernel<<<thread_blocks, threads_per_block>>>(entries_,num_entries_);
    cudaDeviceSynchronize();
    printCUDAError();

    printf("initHeapKernel\n");
    thread_blocks = (heap_size_ + threads_per_block - 1) / threads_per_block;
    initHeapKernel<<<thread_blocks, threads_per_block>>>(heap_, heap_size_,num_buckets_);
    cudaDeviceSynchronize();
    printCUDAError();

    printf("initVoxelsKernel\n");
    initVoxelsKernel<<<vsize/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(voxels_,vsize);
    cudaDeviceSynchronize();
    printCUDAError();

    cudaDeviceSynchronize();

    HashEntry *e=new HashEntry[num_entries_];
    cudaMemcpy(e, entries_, sizeof(HashEntry)*num_entries_, cudaMemcpyDeviceToHost);

    for(int i=0;i<num_entries_;i++)
    {
        printf("E:%d\n",e[i].next_ptr);
    }
//    exit(0);

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
        entries[i].next_ptr = kFreeEntry;
        entries[i].position = make_int3(0, 0, 0);
    }
}

__global__ void initHeapKernel(Heap *heap,int heap_size_, int num_buckets)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (index == 0)
    {
        heap->heap_counter_ = heap_size_ - 1;
    }

    for (int i = index; i < heap_size_; i += stride)
    {
        //heap->heap_[i] = num_blocks - i - 1;
        //heap->heap_[i] = heap_size_ - i + num_buckets -1;
        heap->heap_[i] = i + num_buckets ;
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
