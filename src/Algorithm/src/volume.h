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
    //uint3 resolution;
    //float3 dimensions;

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
        int3 _offset;
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
            _offset=make_int3(0,0,0);

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
          /*
          if (position_local.x < 0)
              position_local.x += block_size;

          if (position_local.y < 0)
              position_local.y += block_size;

          if (position_local.z < 0)
              position_local.z += block_size;
          */
          return position_local;
        }

        bool isNull() const
        {
            return false;
        }

        __host__ __device__ uint3 getResolution() const
        {
            return _resolution;
        }

        __host__ __device__ float3 getVoxelSize() const
        {
            return voxelSize;
        }

        __host__ __device__ int3 getOffset() const
        {
            return _offset;
        }

        __host__ __device__ float3 getOffsetPos() const
        {
            return make_float3(_offset.x*voxelSize.x,
                               _offset.y*voxelSize.y,
                               _offset.z*voxelSize.z);
        }

        __host__ __device__ float3 getDimWithOffset() const
        {
            int3 v=maxVoxel();
            float3 ret;
            ret.x=(v.x)*voxelSize.x;
            ret.y=(v.y)*voxelSize.y;
            ret.z=(v.z)*voxelSize.z;
            return ret;
        }

        __host__ __device__ float3 center() const
        {
            return make_float3(float(_resolution.x)*voxelSize.x*0.5+float(_offset.x)*voxelSize.x,
                               float(_resolution.y)*voxelSize.x*0.5+float(_offset.y)*voxelSize.y,
                               float(_resolution.z)*voxelSize.x*0.5+float(_offset.z)*voxelSize.z);
        }

        __host__ __device__ void addOffset(int3 off)
        {
            _offset.x+=off.x;
            _offset.y+=off.y;
            _offset.z+=off.z;
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
                       hashTable.heap_->heap_,
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
        float vs(const int3 & pos) const
        {
            int blockIdx=-1;
            voxel_t *v=getVoxel(pos,blockIdx);
            if(v==nullptr)
            {
                return 1;
            }
            return v->getTsdf();
        }

        //Get Color data
        __device__
        float3 getColor(int x, int y, int z) const
        {
            int blockIdx=-1;
            int3 pos=make_int3(x,y,z);
            tsdfvh::Voxel *v=getVoxel(pos,blockIdx);
            if(v==nullptr)
            {
                return make_float3(0.0, 0.0, 0.0);
            }
            return v->color;
        }

        __device__
        float3 getColor(const int3 & pos) const
        {
            return getColor(pos.x, pos.y, pos.z);
        }

        __device__
        float3 getColor(const uint3 & pos) const
        {
            return getColor(pos.x, pos.y, pos.z);
        }       


        __device__
        float red(const int3 & pos) const
        {
            return getColor(pos.x, pos.y, pos.z).x;
        }

        __device__
        float green(const int3 & pos) const
        {
            return getColor(pos.x, pos.y, pos.z).y;
        }

        __device__
        float blue(const int3 & pos) const
        {
            return getColor(pos.x, pos.y, pos.z).z;
        }


        __device__
        float3 pos(const int3 & p) const
        {
            return make_float3( ( (p.x + 0.5f) * voxelSize.x),
                                ( (p.y + 0.5f) * voxelSize.y),
                                ( (p.z + 0.5f) * voxelSize.z));
        }

        __device__
        float3 pos(const uint3 & p) const
        {
            return make_float3( ( (p.x + 0.5f) * voxelSize.x),
                                ( (p.y + 0.5f) * voxelSize.y),
                                ( (p.z + 0.5f) * voxelSize.z));
        }

        __device__
        float interp(const float3 & pos) const
        {
            const Fptr fp = &Volume::vs;
            return generic_interp(pos,fp) ;
        }

        __forceinline__ __device__
        tsdfvh::Voxel getVoxelInterp(const float3 &pos,int &blockIdx,bool useColor=true) const;
        
//        __device__
//        float3 rgb_interp(const float3 &p) const
//        {
//            float3 pos=p;

//            float3 rgb;
//            const Fptr red_ptr = &Volume::red;
//            rgb.x=generic_interp(pos,red_ptr);

//            const Fptr green_ptr = &Volume::green;
//            rgb.y=generic_interp(pos,green_ptr);

//            const Fptr blue_ptr = &Volume::blue;
//            rgb.z=generic_interp(pos,blue_ptr);
//            return rgb;
//        }

        __device__
        float generic_interp(const float3 & pos,const Fptr fp) const;

        __device__ float3 grad(const float3 & pos) const;

        void init()
        {
            hashTable.Init(num_of_buckets,
                           bucket_size,
                           block_size);
            clearData();
            printCUDAError();
        }

        void clearData()
        {
            hashTable.setEmpty();
        }
        
        void initDataFromCpu(const VolumeCpu &volCpu)
        {
            num_of_buckets=volCpu.num_of_buckets;
            bucket_size=volCpu.bucket_size;
            block_size=volCpu.block_size;

            hashTable.Init(num_of_buckets,
                           bucket_size,
                           block_size);

            cudaMemcpy((void*)hashTable.entries_,
                       volCpu.entries,
                       hashTable.num_entries_*sizeof(tsdfvh::HashEntry),
                       cudaMemcpyHostToDevice );

            cudaMemcpy((void*)hashTable.heap_->heap_,
                       volCpu.heap,
                       hashTable.heap_size_*sizeof(uint),
                       cudaMemcpyHostToDevice );

            cudaMemcpy((void*)hashTable.voxels_,
                       volCpu.voxels,
                       hashTable.num_entries_*block_size*block_size*block_size*sizeof(tsdfvh::Voxel),
                       cudaMemcpyHostToDevice );

            //hashTable.setEmpty();
        }

        void getCpuData(VolumeCpu &v)
        {
            v.block_size=block_size;
            v.num_of_buckets=num_of_buckets;
            v.bucket_size=bucket_size;

            v.entries=new tsdfvh::HashEntry[hashTable.num_entries_];
            v.heap=new uint[hashTable.heap_size_];
            v.voxels=new tsdfvh::Voxel[hashTable.num_entries_*block_size*block_size*block_size];

            cudaMemcpy(v.entries,
                       (void*)hashTable.entries_,
                       hashTable.num_entries_*sizeof(tsdfvh::HashEntry),
                       cudaMemcpyDeviceToHost );
            cudaMemcpy(v.heap,
                       (void*)hashTable.heap_->heap_,
                       hashTable.heap_size_*sizeof(uint),
                       cudaMemcpyDeviceToHost );
            cudaMemcpy(v.voxels,
                       (void*)hashTable.voxels_,
                       hashTable.num_entries_*block_size*block_size*block_size*sizeof(tsdfvh::Voxel),
                       cudaMemcpyDeviceToHost );
        }
        
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

        __host__ __device__ int3 minVoxel() const
        {
            return _offset;
        }

        __host__ __device__ int3 maxVoxel() const
        {
            return make_int3( int(_resolution.x)+_offset.x,
                              int(_resolution.y)+_offset.y,
                              int(_resolution.z)+_offset.z);
        }

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
