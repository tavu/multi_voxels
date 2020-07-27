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

//__device__ inline
//HashEntry HashTable::GetHashEntry(int i)
//{
//    return entries_[i];
//}

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
        volatile HashEntry &entry=entries_[idx];
        //Block is found
        if( isEqual(entry,position) )
        {
            return idx;
        }        
        //Block has a next pointer
        if(entry.next_ptr >= 0)
        {
            idx=entry.next_ptr;
        }
        else //Block is free tail or locked
        {
            if(isTail(entry) )
            {
                int mutex = atomicCAS( (int*)  &entry.next_ptr, kTailEntry, kLockEntry);
                if (mutex == kTailEntry)
                {
                    int next_ptr=heap_->Consume();

                    entries_[next_ptr].position.x = position.x;
                    entries_[next_ptr].position.y = position.y;
                    entries_[next_ptr].position.z = position.z;

                    entries_[next_ptr].next_ptr = kTailEntry;

                    __threadfence();
                    entries_[idx].next_ptr=next_ptr;
                    __threadfence();

                    return next_ptr;
                }
            }
            else if(isEmpty(entry))
            {
                int mutex = atomicCAS( (int*) &entry.next_ptr, kFreeEntry, kLockEntry);
                if (mutex == kFreeEntry)
                {

                    entry.position.x = position.x;
                    entry.position.y = position.y;
                    entry.position.z = position.z;

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
        //Block is found
        if( isEqual(entries_[idx],position) )
        {
            return idx;
        }

        idx=entries_[idx].next_ptr;
    } while(idx>=0 );

    return -1;
}

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

}  // namespace tsdfvh
