#ifndef HASH_ENTRY_H
#define HASH_ENTRY_H

#include "cuda_runtime.h"

#define kFreeEntry -1
#define kLockEntry -2
#define kTailEntry -3

namespace tsdfvh
{

/**
 * @brief      Struct that represents a hash entry
 *             Hash entries form a linked.
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

        __device__ inline bool isEmpty() const
        {
            return next_ptr==kFreeEntry;
        }

        __device__ inline bool isValid() const
        {
            return next_ptr>0 || next_ptr == kTailEntry;
        }

        __device__ inline bool isLocked() const
        {
            return next_ptr==kLockEntry;
        }

        __device__ inline bool isTail() const
        {
            return next_ptr==kTailEntry;
        }

        __device__ inline bool isEqual(const int3 &pos) const
        {
            return isValid() &&
                   position.x==pos.x &&
                   position.y==pos.y &&
                   position.z==pos.z;
        }

};

}  // namespace tsdfvh

#endif
