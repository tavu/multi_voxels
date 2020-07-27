#include "tsdfvh/hash_table.h"
#include <iostream>
#include "utils.h"
#define THREADS_PER_BLOCK 512

namespace tsdfvh
{
__global__ void initEntriesKernel(HashEntry *entries, int num_entries);

__global__ void initHeapKernel(Heap *heap,
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

    printf("block_size:%d\n",block_size);
    printf("num_entries_:%d\n",num_entries_);
    printf("heap_size_:%d\n",heap_size_);
    printf("vsize:%d\n",vsize);

    cudaMalloc( &entries_, sizeof(HashEntry)*num_entries_);
    cudaMalloc( &voxels_, sizeof(Voxel)*vsize);

    cudaMallocManaged(&heap_, sizeof(Heap));
    heap_->Init(heap_size_);
}

void HashTable::setEmpty()
{
    int threads_per_block = THREADS_PER_BLOCK;
    int thread_blocks = (num_entries_ + threads_per_block ) / threads_per_block;

    int vsize=block_size_*block_size_*block_size_*num_entries_;

    printf("initEntriesKernel\n");
    initEntriesKernel<<<(num_entries_+THREADS_PER_BLOCK)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(entries_,num_entries_);
    printCUDAError();

    printf("initHeapKernel\n");
    thread_blocks = (heap_size_ + threads_per_block - 1) / threads_per_block;
    initHeapKernel<<<thread_blocks, threads_per_block>>>(heap_, heap_size_,num_buckets_);
    printCUDAError();

    printf("initVoxelsKernel\n");
    initVoxelsKernel<<<(vsize+THREADS_PER_BLOCK)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(voxels_,vsize);
    printCUDAError();

//    voxel_t *v=new voxel_t[vsize];
//    cudaMemcpy(v,voxels_,vsize*sizeof(voxel_t),cudaMemcpyDeviceToHost);

//    for(int i=0;i<vsize;i++)
//    {
//        printf("(%d,%d) (%f,%f,%f)\n",v[i].sdf,v[i].weight,v[i].color.x,v[i].color.y,v[i].color.z );
//    }

}

void HashTable::Free()
{
    cudaFree(entries_);
    cudaFree( (void*)voxels_);
    cudaFree(heap_);
}

__global__ void initEntriesKernel(HashEntry *entries, int num_entries)
{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//    for (int i = index; i < num_entries; i += stride)
//    {
//        entries[i].next_ptr = kFreeEntry;
//        entries[i].position = make_int3(0, 0, 0);
//    }
     int idx = blockIdx.x*blockDim.x + threadIdx.x;
     if(idx<num_entries)
     {
         entries[idx].next_ptr = kFreeEntry;
         entries[idx].position = make_int3(0, 0, 0);
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

__global__ void initVoxelsKernel(voxel_t *voxels, int size )
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size)
    {
//        printf("IDX:%d\n",idx);
        //=make_float3(0.0,0.0,0.0);
        voxels[idx].color.x=0.0;
        voxels[idx].color.y=0.0;
        voxels[idx].color.z=0.0;

        voxels[idx].setWeight(0.0);
        voxels[idx].setTsdf(1.0);
    }
}

}  // namespace tsdfvh
