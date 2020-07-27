#include "tsdfvh/hash_table.h"
#include"tsdfvh/voxel.h"
#include<cuda_runtime.h>
#include <iostream>
#include"utils.h"

#define THREADS_PER_BLOCK 512

namespace tsdfvh 
{

__host__ __device__ inline
int HashTable::GetNumEntries() {
    return num_entries_;
}

__device__ inline
HashEntry HashTable::GetHashEntry(int i)
{
    return entries_[i];
}

__device__
inline voxel_t& HashTable::GetVoxel(int entry_idx, int3 vpos) const
{
    int vidx=vpos.x + vpos.y * block_size_ + vpos.z * block_size_ * block_size_;
    int idx=entry_idx*block_size_*block_size_*block_size_+vidx;
    return voxels_[idx];
}


__device__ inline
int HashTable::AllocateBlock(const int3 &position)
{
#ifdef __CUDACC__
    int idx = Hash(position);
    do
    {
        HashEntry &entry=entries_[idx];
        //Block is found
        if( entry.isEqual(position) )
        {
            //printf("block found.\n");
            return idx;
        }

        //Block has a next pointer
        if(entry.next_ptr >= 0)
        {
            idx=entry.next_ptr;
        }
        else //Block is free tail or locked
        {
            if(entry.isTail() )
            {
                int mutex = atomicCAS(&entry.next_ptr, kTailEntry, kLockEntry);
                if (mutex == kTailEntry)
                {
                    int next_ptr=heap_->Consume();

                    entries_[next_ptr].position = position;
                    entries_[next_ptr].next_ptr = kTailEntry;

                    __threadfence();
                    entries_[idx].next_ptr=next_ptr;
                    __threadfence();
                    return next_ptr;
                }
            }
            else if(entry.isEmpty() )
            {
                int mutex = atomicCAS(&entry.next_ptr, kFreeEntry, kLockEntry);
                if (mutex == kFreeEntry)
                {
                    entry.position = position;

                     __threadfence();
                    entries_[idx].next_ptr=kTailEntry;
                    __threadfence();
                    return idx;
                }
            }
            /*
            if(entry.isLocked() )
            {
                unsigned int ns = 8;
                __nanosleep(ns);
            }
            */
        }
    } while(idx>=0);
#endif
    return -1;
}


//TODO
__device__ inline
int HashTable::DeleteBlock(const int3 &position)
{
    return -1;
}

__device__ inline
int HashTable::FindHashEntry(int3 position) const
{
    int idx = Hash(position);
    do
    {
        HashEntry &entry=entries_[idx];
        //Block is found
        if( entry.isEqual(position) )
        {
            return idx;
        }

        idx=entry.next_ptr;
    } while(idx>=0 );

    return -1;
}

/*
__device__ inline
int HashTable::AllocateBlock(const int3 &position)
{
#ifdef __CUDACC__

    int bucket_idx = Hash(position);
//    printf("Bucket idx:%d\n",bucket_idx);


    int free_entry_idx = -1;
    for (int i = 0; i < bucket_size_; i++) 
    {
        if (entries_[bucket_idx + i].position.x == position.x &&
            entries_[bucket_idx + i].position.y == position.y &&
            entries_[bucket_idx + i].position.z == position.z &&
            entries_[bucket_idx + i].pointer != kFreeEntry) 
        {
            printf("block found.\n");
            return 0;
        }
        
        if (free_entry_idx == -1 &&
            entries_[bucket_idx + i].pointer == kFreeEntry) 
        {
            free_entry_idx = bucket_idx + i;
        }
    }

    if (free_entry_idx != -1) 
    {
        int mutex = 0;
        mutex = atomicCAS(&entries_[free_entry_idx].pointer, kFreeEntry, kLockEntry);
        
        if (mutex == kFreeEntry) 
        {
            entries_[free_entry_idx].position = position;
            entries_[free_entry_idx].pointer = heap_->Consume();
            atomicAdd(&num_allocated_blocks_, 1);
            return 1;
        }
    }

#endif
    printf("done...\n");
    return -1;
}



__device__ inline
int HashTable::AllocateBlock(const int3 &position)
{
#ifdef __CUDACC__
    int bucket_idx = Hash(position);

    int free_entry_idx = -1;
    for (int i = 0; i < bucket_size_; i++)
    {
        if (entries_[bucket_idx + i].position.x == position.x &&
            entries_[bucket_idx + i].position.y == position.y &&
            entries_[bucket_idx + i].position.z == position.z &&
            entries_[bucket_idx + i].pointer != kFreeEntry)
        {
            return 0;
        }

        if (free_entry_idx == -1 &&
            entries_[bucket_idx + i].pointer == kFreeEntry)
        {
            free_entry_idx = bucket_idx + i;
        }
    }

    if (free_entry_idx != -1)
    {
        int mutex = 0;
        mutex = atomicCAS(&entries_[free_entry_idx].pointer, kFreeEntry, kLockEntry);

        if (mutex == kFreeEntry)
        {
            entries_[free_entry_idx].position = position;
            entries_[free_entry_idx].pointer = heap_->Consume();
            //atomicAdd(&num_allocated_blocks_, 1);
            return 1;
        }
    }
#endif
    return -1;
}

//TODO
__device__ inline
bool HashTable::DeleteBlock(const int3 &position)
{
    int bucket_idx = Hash(position);

    for (int i = 0; i < bucket_size_; i++) 
    {
        if (entries_[bucket_idx + i].position.x == position.x &&
            entries_[bucket_idx + i].position.y == position.y &&
            entries_[bucket_idx + i].position.z == position.z &&
            entries_[bucket_idx + i].pointer != kFreeEntry) 
        {
            int ptr = entries_[bucket_idx + i].pointer;
            for(int j=0;j<block_size_ * block_size_ * block_size_; j++) 
            {
//                voxel_blocks_[ptr].at(j).sdf = 0;
//                voxel_blocks_[ptr].at(j).color = make_float3(0, 0, 0);
//                voxel_blocks_[ptr].at(j).weight = 0;
            }
            
            heap_->Append(ptr);
            entries_[bucket_idx + i].pointer = kFreeEntry;
            entries_[bucket_idx + i].position = make_int3(0, 0, 0);
            
            return true;
        }
    }
    return false;
}

__host__ __device__ inline
HashEntry HashTable::FindHashEntry(int3 position) const
{
    int bucket_idx = Hash(position);
    for (int i = 0; i < bucket_size_; i++) 
    {
        if (entries_[bucket_idx + i].position.x == position.x &&
            entries_[bucket_idx + i].position.y == position.y &&
            entries_[bucket_idx + i].position.z == position.z &&
            entries_[bucket_idx + i].pointer != kFreeEntry) 
        {
            return entries_[bucket_idx + i];
        }
    }
    
    HashEntry entry;
    entry.position = position;
    entry.pointer = kFreeEntry;
    return entry;
}
*/
__host__ __device__ inline
int HashTable::Hash(int3 position) const
{
    const int p1 = 73856093;
    const int p2 = 19349669;
    const int p3 = 83492791;

    int result = ((position.x * p1) ^ (position.y * p2) ^ (position.z * p3)) % num_buckets_;

    if (result < 0) 
    {
        result += num_buckets_;
    }
    return result * bucket_size_;
}

//inline int HashTable::GetNumAllocatedBlocks()
//{
//    return num_allocated_blocks_;
//}


}  // namespace tsdfvh
