#ifndef VOLUME_H
#define VOLUME_H

#include"utils.h"
#include"cutil_math.h"
#include"kparams.h"
#include<iostream>
#include"sMatrix.h"

#include"kparams.h"

#include"tsdfvh/voxel.h"
#include"tsdfvh/hash_table.h"
#include"tsdfvh/hash_entry.h"

#include"utils.h"
#include"tsdfvh/hash_entry.h"
#include"tsdfvh/hash_entry.h"

/**
 * @brief      Struct storing a map on RAM
 */
struct VolumeCpu
{
    /** frame id */
    uint frame;
    /** pose of key frame id */
    sMatrix4 pose;

    /** bucket size */
    int bucket_size;
    /** number of buckets */
    int num_of_buckets;
    /** size of block (cubical) */
    int block_size;

    /** voxels */
    tsdfvh::Voxel *voxels;
    /** hash table entries */
    tsdfvh::HashEntry *entries;
    /** heap */
    uint *heap;
};


/**
 * @brief      Class representing the map.
 */
class Volume
{
    private:
        typedef float (Volume::*Fptr)(const int3&) const;

        uint3 _resolution;
        float3 dim;
        float3 voxelSize;
        tsdfvh::HashTable hashTable;
        int block_size;
        int bucket_size;
        int num_of_buckets;


    public:
        /**
        * @brief      Construct the volume
        *
        * @param[in]  params The input parameters.
        */
        Volume(const kparams_t &params)
        {
            _resolution = params.volume_resolution;
            dim = params.volume_size;

            voxelSize=dim/_resolution;

            block_size=params.block_size;
            bucket_size=params.bucket_size;
            num_of_buckets=params.num_buckets;

        }

        /**
        * @brief      Gets the block size
        *
        * @return     Returns the size of block (cubical)
        */
        __host__ __device__
        int getBlockSize() const
        {
            return block_size;
        }


        /**
        * @brief      Gets the buckets size
        *
        * @return     Returns the size of buckets
        */
        __host__ __device__
        int getBucketSize() const
        {
            return bucket_size;
        }


        /**
        * @brief      Gets the number of buckets.
        *
        * @return     Returns the number of buckets.
        */
        __host__ __device__
        int getNumOfBuckets() const
        {
            return num_of_buckets;
        }

        /**
        * @brief      Gets the position of the bucket from x,y,z coordinates of a voxel.
        * @param[in]  x The x coordinate of a voxel
        * @param[in]  y The y coordinate of a voxel
        * @param[in]  z The z coordinate of a voxel
        * @return     Returns the position of the buckets.
        */
        __host__ __device__
        int3 blockPosition(int x, int y, int z) const
        {
            return make_int3(x / block_size,
                             y / block_size,
                             z / block_size);
        }

        /**
        * @brief      Gets the position of a voxel inside a bucket
        * @param[in]  x The x coordinate of a voxel
        * @param[in]  y The y coordinate of a voxel
        * @param[in]  z The z coordinate of a voxel
        * @return     Returns the position of the voxel inside the bucket.
        */
        __host__ __device__
        int3 voxelPosition(int x, int y, int z) const
        {
          int3 position_local = make_int3(x % block_size,
                                          y % block_size,
                                          z % block_size);
          return position_local;
        }

        /**
        * @brief      Gets resolution of the map
        * @return     Returns the resolution of the map
        */
        __host__ __device__ uint3 getResolution() const
        {
            return _resolution;
        }

        /**
        * @brief      Gets the size of the voxels.
        * @return     Returns the size of the voxels.
        */
        __host__ __device__ float3 getVoxelSize() const
        {
            return voxelSize;
        }

        /**
        * @brief      Gets the dimension of the map in meters
        * @return     Returns the dimension of the map in meters
        */
        __host__ __device__ float3 getDimensions() const
        {
            return dim;
        }

#ifdef __CUDACC__

        /**
        * @brief      Gets the hash entry block given its index.
        * @param[in]  blockIdx The index of the block
        * @return     Returns the HashEntry
        */
        __device__ __forceinline__
        tsdfvh::HashEntry getHashEntry(int blockIdx) const
        {
            tsdfvh::HashEntry ret;
            ret.next_ptr=hashTable.entries_[blockIdx].next_ptr;
            ret.position=make_int3(hashTable.entries_[blockIdx].position.x,
                                   hashTable.entries_[blockIdx].position.y,
                                   hashTable.entries_[blockIdx].position.z);
            return ret;
        }

        /**
        * @brief          Insert a voxel into map.
        * @param[in]      pos The world coordinates of the voxel.
        * @param[in|out]  block_idx A hint for the block index.
        *                 If block_idx points into the block that contains the voxel
        *                 then the voxel is inserted into the proper position inside the
        *                 block avoiding searching the hash table.
        *                 Otherwise the hash table is searched and the block_idx is set according to the final
        *                 block index.
        *                 If there is already a voxel at the given position then a reference
        *                 to this voxel is returned.
        * @return         The newly inserted voxel.
        */
        __device__ __forceinline__
        voxel_t* insertVoxel(const int3 &pos, int &block_idx)
        {
            int3 block_position = blockPosition(pos.x,pos.y,pos.z);
            int3 local_voxel = voxelPosition(pos.x,pos.y,pos.z);

            if(block_idx>=0 && isEqual(hashTable.entries_[block_idx],block_position) )
            {
                return &hashTable.GetVoxel(block_idx,local_voxel);
            }

            block_idx=hashTable.AllocateBlock(block_position);
            if(block_idx>=0)
                return &hashTable.GetVoxel(block_idx,local_voxel);
            return nullptr;
        }


        /**
        * @brief          Gets a voxel into map.
        * @param[in]      pos The position of the voxel on the map.
        * @param[in|out]  block_idx A hint for the block index.
        *                 If block_idx points into the block that contains the voxel
        *                 then this function return the proper reference avoid searching the hash table.
        *                 Otherwise the hash table is searched and the block_idx is set according to the final
        *                 block index.
        *                 If there is no voxel at the given position then nullptr is returned and block_idx is set to a negative value.
        * @return         The requested voxel.
        */
        __device__ inline
        voxel_t* getVoxel(const int3 &pos, int &block_idx) const
        {
            if(block_idx<0 || !isEqual(hashTable.entries_[block_idx],pos) )
            {
                int3 block_position = blockPosition(pos.x,pos.y,pos.z);
                block_idx=hashTable.FindHashEntry(block_position);
            }

            if (block_idx<0)
            {
                return nullptr;
            }
            int3 local_voxel = voxelPosition(pos.x,pos.y,pos.z);
            return &hashTable.GetVoxel(block_idx,local_voxel);
        }

        /**
        * @brief          Gets the tsdf value of a voxel at given position
        * @param[in]      pos The position of the voxel on the map.
        * @return         The tsdf value of a voxel at given position
        */
        __device__ inline
        float vs(const int3 &pos) const
        {
            int blockIdx=-1;
            voxel_t *v=getVoxel(pos,blockIdx);
            if(v==nullptr)
            {
                return 1;
            }
            return v->getTsdf();
        }

        /**
        * @brief          Gets the world position of a voxel in map position pos.
        * @param[in]      pos The position of the voxel on the map.
        *
        * @return         The coordinates on the world.
        */
        __device__
        float3 pos(const int3 & p) const
        {
            return make_float3( ( (p.x + 0.5f) * voxelSize.x),
                                ( (p.y + 0.5f) * voxelSize.y),
                                ( (p.z + 0.5f) * voxelSize.z));
        }

        /**
        * @brief          Gets a voxel type with values created from interpolation.
        *                 Tsdf is calculated from interpolation.
        *                 weight is the average weight of the neighboring voxels.
        *                 if useColor is True this function also calculates the color of the voxels
        *                 by interpolating the neighboring voxels.
        *
        * @param[in]      pos The position of interpolation
        * @param[in]      blockIdx A hint for the block index.
        * @param[out]     out The output data
        * @param[out]     useColor True for calculating the voxel's color.
        *
        * @return         The number of neighboring voxels that are empty.
        */
        __forceinline__ __device__
        int getVoxelInterp(const float3 &pos,
                           int &blockIdx,
                           tsdfvh::Voxel &out,
                           bool useColor=true) const;

        /**
        * @brief          Gets the gradient of the given position.
        *
        * @param[in]      pos A position in the world.
        * @return         The gradient.
        */
        __device__ float3 grad(const float3 & pos) const;
#endif

        /**
        * @brief          Allocates memory and sets the map as empty.
        */
        void init()
        {
            //allocate hash memory
            hashTable.Init(num_of_buckets,
                           bucket_size,
                           block_size);

            //Set hash table as empty
            clearData();
            printCUDAError();
        }

        /**
        * @brief          Sets the map as empty.
        *                 Data has to be allocated first.
        */

        void clearData()
        {
            hashTable.setEmpty();
        }
        

        /**
        * @brief         Initialize volume from Ram memory
        */
        void initDataFromCpu(const VolumeCpu &volCpu);

        /**
        * @brief         Store volume to RAM.
        * @param[out]     v Volume data in ram
        */
        void getCpuData(VolumeCpu &v);
        
        /**
        * @brief         Test if a given point lay inside the volume
        * @param[in]     pos A position in the world.
        */
        __host__ __device__ __forceinline__ 
        bool isPointInside(const float3 &pos) const
        {
            if( pos.x<0 || pos.x >= getDimensions().x ||
                pos.y<0 || pos.x >= getDimensions().y ||
                pos.z<0 || pos.x >= getDimensions().z)
            {
                return false;
            }
            return true;
        }

        /**
        * @brief  Frees memory
        */
        void release()
        {
            hashTable.Free();
        }

        /**
        * @brief   Gets the number of allocated entries in hash table.
        * @return  Return the number of allocated entries in hash table.
        */
        int getHashSize() const
        {
            return hashTable.num_entries_;
        }

        /**
        * @brief   Store hash data into RAM
        * @return  Return the number of allocated entries in hash table.
        */
        void saveHash(tsdfvh::HashEntry *cpudata) const
        {
            cudaMemcpy(cpudata,
                       (void *)hashTable.entries_,
                       hashTable.num_entries_*sizeof(tsdfvh::HashEntry),
                       cudaMemcpyDeviceToHost);
        }

        int getHeapSize() const
        {
            return hashTable.heap_size_;
        }

        void saveHeap(uint *cpudata) const
        {
            cudaMemcpy(cpudata,
                       hashTable.heap_.heap_,
                       hashTable.heap_size_*sizeof(uint),
                       cudaMemcpyDeviceToHost);
        }
};

//Usefull functions
void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData);
void saveVoxelsToFile(const char *fileName, const uint3 &resolution, float vox_size, const short2 *voxels);


#include"volume_impl.h"

#endif // VOLUME_H
