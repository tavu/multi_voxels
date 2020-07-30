#include "tsdfvh/hash_table.h"
#include"tsdfvh/voxel.h"
#include<cuda_runtime.h>
#include <iostream>
#include"utils.h"

#define THREADS_PER_BLOCK 512

namespace tsdfvh 
{

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
                    int next_ptr=heap_.Consume();

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
bool HashTable::DeleteBlock(const int3 &position)
{
#ifdef __CUDACC__
    int start_idx = Hash(position);
    volatile HashEntry *start_entry=&entries_[start_idx];
    int start_ptr;
    int mutex;

    //lock the start position of the linked list
    do
    {
        start_ptr=start_entry->next_ptr;
        if(start_ptr==kFreeEntry)
        {
            return false;
        }
        mutex = atomicCAS( (int*) &start_entry->next_ptr, start_ptr, kLockEntry);
    }while(mutex != start_ptr);

    //only one entry on the list
    if(start_ptr==kTailEntry)
    {
        if(start_entry->position.x==position.x &&
           start_entry->position.y==position.y &&
           start_entry->position.z==position.z)
        {
            start_entry->position.x=0;
            start_entry->position.y=0;
            start_entry->position.z=0;

            for(int i=0;i<block_size_*block_size_*block_size_;i++)
            {
                voxels_[start_idx+i].setTsdf(1.0);
                voxels_[start_idx+i].setWeight(0.0);
                voxels_[start_idx+i].setColor(make_float3(0.0, 0.0, 0.0));
            }
            __threadfence();
            start_entry->next_ptr = kFreeEntry;
            __threadfence();
            return true;
        }
        else
        {
            start_entry->next_ptr = start_ptr;
            __threadfence();
            return false;
        }
    }
    //The first entry has to be removed.
    //Replace it with tail
    else if(start_entry->position.x==position.x &&
            start_entry->position.y==position.y &&
            start_entry->position.z==position.z)
    {
        int idx=start_entry->next_ptr;
        volatile HashEntry *prev_entry=start_entry;
        volatile HashEntry *entry=&entries_[idx];
        while(entry->next_ptr!=kTailEntry)
        {
            prev_entry=entry;
            idx=entry->next_ptr;
            entry=&entries_[idx];
        }

        prev_entry->next_ptr=kTailEntry;

        start_entry->position.x=entry->position.x;
        start_entry->position.y=entry->position.y;
        start_entry->position.z=entry->position.z;

        for(int i=0;i<block_size_*block_size_*block_size_;i++)
        {
            voxels_[start_idx+i].sdf=voxels_[idx+i].sdf;
            voxels_[start_idx+i].weight=voxels_[idx+i].weight;
            voxels_[start_idx+i].r=voxels_[idx+i].r;
            voxels_[start_idx+i].g=voxels_[idx+i].g;
            voxels_[start_idx+i].b=voxels_[idx+i].b;

            voxels_[idx+i].setTsdf(1.0);
            voxels_[idx+i].setWeight(0.0);
            voxels_[idx+i].setColor(make_float3(0.0, 0.0, 0.0));
        }

        __threadfence();
        start_entry->next_ptr = start_ptr;
        __threadfence();
        return true;

    }
    else //Some other entry has to be removed
    {
        int idx=start_entry->next_ptr;
        volatile HashEntry *prev_entry=start_entry;
        volatile HashEntry *entry=&entries_[idx];
        bool found=false;
        while(entry->next_ptr!=kTailEntry )
        {
            if(entry->position.x==position.x &&
               entry->position.y==position.y &&
               entry->position.z==position.z)
            {
                found=true;
                break;
            }
            prev_entry=entry;
            idx=entry->next_ptr;
            entry=&entries_[idx];
        }
        if(!found)
        {
            start_entry->next_ptr = start_ptr;
            __threadfence();
            return false;
        }

        prev_entry->next_ptr=entry->next_ptr;
        entry->position.x=0;
        entry->position.y=0;
        entry->position.z=0;
        entry->next_ptr = kFreeEntry;

        for(int i=0;i<block_size_*block_size_*block_size_;i++)
        {
            voxels_[idx+i].setTsdf(1.0);
            voxels_[idx+i].setWeight(0.0);
            voxels_[idx+i].setColor(make_float3(0.0, 0.0, 0.0));
        }
        __threadfence();
         start_entry->next_ptr = start_ptr;
         __threadfence();
        return true;
    }
#endif
    return false;
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
