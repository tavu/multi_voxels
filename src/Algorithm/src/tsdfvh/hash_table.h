#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <cuda_runtime.h>
#include <stdio.h>
#include "hash_entry.h"
#include "heap.h"
#include "voxel.h"

namespace tsdfvh
{

/**
 * @brief      Class for handling the hash table. It takes care of allocating
 *             and deleting voxel blocks and handles the hash entries. The
 *             unit used for coordinates is a voxel block.
 */
class HashTable
{
    public:
        /**
        * @brief      Initializes the members of the class and allocates the memory
        *             for the hash table and the voxel grid.
        *
        * @param[in]  num_buckets  The number of buckets
        * @param[in]  bucket_size  The size of a bucket
        * @param[in]  block_size   The size in voxels of a side of a voxel block
        */
        void Init(int num_buckets, int bucket_size, int block_size);

        /**
        * @brief      Frees the memory allocated by the class.
        */
        void Free();

        /**
        * @brief      Sets hash table as empty.
        */
        void setEmpty();
        /**
        * @brief      Allocates a voxel block.
        *
        * @param[in]  position  The 3D position of the voxel block
        *
        * @return     1 if the block was successfully allocated, 0 if the voxel block
        *             in that position was already allocated, -1 if the voxel block
        *             cannot be allocated (bucket full).
        */
        __device__ int AllocateBlock(const int3 &position);

        /**
        * @brief      Deletes a voxel block.
        *
        * @param[in]  position  The 3D position of the voxel block
        *
        * @return     True if the block was successfully deleted. False if the block
        *             was not found.
        */
        __device__ int DeleteBlock(const int3 &position);

        /**
        * @brief      Returns the hash entry corresponding to a given position.
        *
        * @param[in]  position  The 3D position of the entry
        *
        * @return     The hash entry corresponding to the position.
        */
        __device__ __forceinline__
        int FindHashEntry(int3 position) const;

        /**
        * @brief      Returns the number of allocated blocks.
        *
        * @return     The number of allocated blocks.
        */
        __forceinline__ int GetNumAllocatedBlocks();

        /**
        * @brief      Gets the number of entries.
        *
        * @return     The number of entries.
        */
        __host__ __device__ __forceinline__ int GetNumEntries();

        /**
        * @brief      Gets the hash entry at the given index.
        *
        * @param[in]  i     The index of the hash entry
        *
        * @return     The hash entry.
        */
//        __device__ __forceinline__ HashEntry GetHashEntry(int i);

        __device__
        inline voxel_t& GetVoxel(int entry_idx, int3 vpos) const;

        /**
        * @brief      Computes the hash value from a 3D position.
        *
        * @param[in]  position  The 3D position
        *
        * @return     The hash value.
        */
        __host__ __device__ int Hash(int3 position) const;

        /** Entries of the hash table */
        volatile HashEntry *entries_;

        /** Voxels in the grid */
        voxel_t *voxels_;

        /** Object that handles the indices of the voxel blocks */
        Heap heap_;

        /** Total maximum number of heap */
        int heap_size_;

        /** Total number of buckets in the table */
        int num_buckets_;

        /** Size of a bucket */
        int bucket_size_;

        /** Total number of entries in the table (num_buckets_ * bucket_size_) */
        int num_entries_;

        /** Size in voxels of the side of a voxel block */
        int block_size_;

//        /** Number of blocks currently allocated */
//        int num_allocated_blocks_;
};

}  // namespace tsdfvh

#include"hash_table.cuh"

#endif //HASH_TABLE_H
