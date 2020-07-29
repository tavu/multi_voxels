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

struct VolumeCpu
{
    uint frame;
    sMatrix4 pose;

    int bucket_size;
    int num_of_buckets;
    int block_size;

    tsdfvh::Voxel *voxels;
    tsdfvh::HashEntry *entries;
    uint *heap;
};



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
        Volume(const kparams_t &params)
        {
            _resolution = params.volume_resolution;
            dim = params.volume_size;

            voxelSize=dim/_resolution;

            block_size=params.block_size;
            bucket_size=params.bucket_size;
            num_of_buckets=params.num_buckets;

        }

        __host__ __device__
        int getBlockSize() const
        {
            return block_size;
        }

        __host__ __device__
        int getBucketSize() const
        {
            return bucket_size;
        }


        __host__ __device__
        int getNumOfBuckets() const
        {
            return num_of_buckets;
        }

        __host__ __device__
        int3 blockPosition(int x, int y, int z) const
        {
            return make_int3(x / block_size,
                             y / block_size,
                             z / block_size);
        }

        __host__ __device__
        int3 voxelPosition(int x, int y, int z) const
        {
          int3 position_local = make_int3(x % block_size,
                                          y % block_size,
                                          z % block_size);
          return position_local;
        }

        __host__ __device__ uint3 getResolution() const
        {
            return _resolution;
        }

        __host__ __device__ float3 getVoxelSize() const
        {
            return voxelSize;
        }

        __host__ __device__ float3 getDimensions() const
        {
            return dim;
        }                

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

        //insert voxel
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

        int getHashSize() const
        {
            return hashTable.num_entries_;
        }

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

        //Get Voxel
        __device__ voxel_t* getVoxel(const int3 &pos, int &block_idx) const
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

        __device__
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

        __device__
        float3 pos(const int3 & p) const
        {
            return make_float3( ( (p.x + 0.5f) * voxelSize.x),
                                ( (p.y + 0.5f) * voxelSize.y),
                                ( (p.z + 0.5f) * voxelSize.z));
        }

        __forceinline__ __device__
        int getVoxelInterp(const float3 &pos,
                           int &blockIdx,
                           tsdfvh::Voxel &out,
                           bool useColor=true) const;

        __device__ float3 grad(const float3 & pos) const;

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

        //Sets hash table as empty
        void clearData()
        {
            hashTable.setEmpty();
        }
        
        //Initialize volume from CPU memory
        void initDataFromCpu(const VolumeCpu &volCpu);

        //Store volume to CPU memory
        void getCpuData(VolumeCpu &v);
        
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

        //free memory
        void release()
        {
            hashTable.Free();
        }
};

//Usefull functions
void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData);
void saveVoxelsToFile(const char *fileName, const uint3 &resolution, float vox_size, const short2 *voxels);


#include"volume_impl.h"

#endif // VOLUME_H
