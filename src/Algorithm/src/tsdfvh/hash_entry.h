#ifndef HASH_ENTRY_H
#define HASH_ENTRY_H

#include "cuda_runtime.h"


#define kFreeEntry -1 //free block
#define kLockEntry -2 //block locked from other thread
#define kTailEntry -3 //This block is valid and also a tail in the linked list

namespace tsdfvh
{

/**
 * @brief      Struct that represents a hash entry
 *             Hash entries form a linked list.
 */

class HashEntry
{
    public:
        /** Entry position (lower left corner of the voxel block) */
        int3 position;

        /**
         * Pointer to the Next entry
         * If next_ptr is kFreeEntry this entry is unused
         * If next_ptr is kNone this entry is the tail of the list
         * If next_ptr is kLockEntry this entry is edited by another thread.
         */
        int next_ptr = kFreeEntry;
};

__device__ inline
bool isEqual(volatile HashEntry &entry, const int3 &position)
{
    return (entry.next_ptr>=0 || entry.next_ptr == kTailEntry) &&
            entry.position.x==position.x &&
            entry.position.y==position.y &&
            entry.position.z==position.z;
}

__device__ inline
bool isTail(volatile HashEntry &entry)
{
    return entry.next_ptr==kTailEntry;
}

__device__ inline
bool isLocked(volatile HashEntry &entry)
{
    return entry.next_ptr==kLockEntry;
}

__device__ inline
bool isEmpty(volatile HashEntry &entry)
{
    return entry.next_ptr==kFreeEntry;
}


}  // namespace tsdfvh

#endif
