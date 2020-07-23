#ifndef VOLUME_H
#define VOLUME_H

#include"cutil_math.h"
#include"kparams.h"
#include<iostream>
#include"sMatrix.h"

#include"kparams.h"

#include"tsdfvh/voxel.h"
#include"tsdfvh/hash_table.h"
#include"tsdfvh/hash_entry.h"
#include"tsdfvh/voxel_block.h"

//for short x
//x * 0.00003051944088f
//float2 ret = make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f
//data[p] = make_short2(d.x * 32766.0f, d.y);
//float2 ret = make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f

#include"utils.h"

struct VolumeCpu
{
    uint frame;
    sMatrix4 pose;
    uint3 resolution;
    float3 dimensions;

    tsdfvh::Voxel *voxels;
};

class Volume
{
    private:
        typedef float (Volume::*Fptr)(const int3&) const;
        const kparams_t &params;
        int block_size;
        int bucket_size;

    public:
        Volume(const kparams_t &par)
            :params(par)
        {
            _resolution = params.volume_resolution;
            dim = params.volume_size;
            voxels = nullptr;

            uint size=_resolution.x * _resolution.y * _resolution.z;
            cudaMalloc((void**)&voxels, size*sizeof(tsdfvh::Voxel));
            voxelSize=dim/_resolution;
            _offset=make_int3(0,0,0);
            block_size=params.block_size;

            block_resolution=make_uint3(_resolution.x/block_size,
                                        _resolution.y/block_size,
                                        _resolution.z/block_size);

            if(_resolution.x%block_size)
                block_resolution.x++;
            if(_resolution.y%block_size)
                block_resolution.y++;
            if(_resolution.z%block_size)
                block_resolution.z++;

            bucket_size=params.bucket_size;
            hashTable.Init(params.num_buckets,
                           params.bucket_size,
                           params.num_blocks,
                           params.block_size);
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

          if (position_local.x < 0)
              position_local.x += block_size;

          if (position_local.y < 0)
              position_local.y += block_size;

          if (position_local.z < 0)
              position_local.z += block_size;

          return position_local;
        }

        bool isNull() const
        {
            return voxels == nullptr;
        }

        __host__ __device__ uint3 getResolution() const
        {
            return _resolution;
        }

        __host__ __device__ float3 getVoxelSize() const
        {
            return voxelSize;
        }

        __host__ __device__ float3 getSizeInMeters() const
        {
            return make_float3(voxelSize.x*_resolution.x,
                               voxelSize.y*_resolution.y,
                               voxelSize.z*_resolution.z);
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
            //float3 ret=make_float3( dim.x+_offset.x*voxelSize.x,
            //                        dim.y+_offset.y*voxelSize.y,
            //                        dim.z+_offset.z*voxelSize.z);

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

        __host__ __device__ tsdfvh::Voxel*  getVoxelsPtr() const
        {
            return voxels;
        }

        __host__ __device__ float3 getDimensions() const
        {
            return dim;
        }

        //Get Voxel
        __device__ __forceinline__
        tsdfvh::Voxel getVoxel(int x, int y, int z) const
        {
            int3 block_position = blockPosition(x,y,z);
            int3 local_voxel = voxelPosition(x,y,z);
            tsdfvh::HashEntry entry = hashTable.FindHashEntry(block_position);

            if (entry.pointer == kFreeEntry)
            {
                tsdfvh::Voxel voxel;
                voxel.sdf = 0;
                voxel.color = make_float3(0.0, 0.0, 0.0);
                voxel.weight = 0;
                return voxel;
            }
//            const tsdfvh::VoxelBlock &vb=hashTable.GetVoxelBlock(entry);
//            int vidx=getIdx(local_voxel.x,
//                            local_voxel.y,
//                            local_voxel.z,
//                            _resolution);

//            int blockIdx=getIdx(entry.position.x,
//                                entry.position.y,
//                                entry.position.z,
//                                block_resolution);

//            int fidx=blockIdx*block_size+vidx;
            tsdfvh::Voxel &voxel=hashTable.GetVoxel(entry,local_voxel);
            return voxel;
        }

        __device__ __forceinline__
        void setVoxel(const tsdfvh::Voxel &v, int x, int y, int z)
        {
            uint idx=getIdx(x,y,z,_resolution);
            //voxels[idx]=v;

            int3 block_position = blockPosition(x,y,z);

            int status=-1;
            int count=0;
            do
            {
                status=hashTable.AllocateBlock(block_position);
                count++;
            }while(status==-1 && count<bucket_size);

            if(status<0)
            {
                printf("Error allocating block\n");
                return ;
            }

            tsdfvh::HashEntry entry = hashTable.FindHashEntry(block_position);

            if(entry.pointer<0)
            {
                printf("Error finding block\n");
                return ;
            }

//            const tsdfvh::VoxelBlock &vb=hashTable.GetVoxelBlock(entry);
//            int blockIdx=getIdx(entry.position.x,
//                                entry.position.y,
//                                entry.position.z,
//                                block_resolution);

            int3 local_voxel = voxelPosition(x,y,z);
//            int vidx=getIdx(local_voxel.x,
//                            local_voxel.y,
//                            local_voxel.z,
//                            _resolution);

//            int fidx=blockIdx*block_size+vidx;
            tsdfvh::Voxel &voxel=hashTable.GetVoxel(entry,local_voxel);
            voxel=v;
        }

//        //IDX
//        __host__ __device__ __forceinline__
//        uint getIdx(int x, int y, int z) const
//        {
//            return x + y * _resolution.x + z * _resolution.x * _resolution.y;
//        }

//        __host__ __device__ __forceinline__
//        uint getIdx(const uint3 &pos) const
//        {
//            return getIdx(pos.x, pos.y, pos.z);
//        }

//        __host__ __device__ __forceinline__
//        uint getIdx(const int3 &pos) const
//        {
//            return getIdx(pos.x, pos.y, pos.z);
//        }


        //Get SDF data
        __device__
        float2 getData(int x, int y, int z) const
        {
            tsdfvh::Voxel v=getVoxel(x, y, z);
            float2 ret=make_float2(v.sdf,
                                  v.weight);

            return ret;
        }

        __device__
        float2 operator[](const int3 & pos) const
        {
            return getData(pos.x,pos.y,pos.z);
        }

        __device__
        float2 operator[](const uint3 & pos) const
        {
            return getData(pos.x,pos.y,pos.z);
        }

        __device__
        float vs(const int3 & pos) const
        {
            return getData(pos.x, pos.y, pos.z).x;
        }

        __device__
        float vw(const int3 & pos) const
        {
            return getData(pos.x, pos.y, pos.z).y;
        }


        __device__
        float vww(const int3 & pos) const
        {
            short w=getData(pos.x, pos.y, pos.z).y;
            if(w>0)
                return 1.0;
            return 0.0;
        }


        //Get Color data
        __device__
        float3 getColor(int x, int y, int z) const
        {
            tsdfvh::Voxel v=getVoxel(x, y, z);
            return v.color;
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

        //Set Data
        __device__
        void set(int x, int y, int z, const float2 &d, const float3 &c)
        {
            tsdfvh::Voxel v;
            v.color=c;
            v.sdf=d.x;
            v.weight=d.y;
            setVoxel(v, x, y, z);
        }

        __device__
        void set(const int3 & pos, const float2 & d)
        {
            float3 c=make_float3(0.0,0.0,0.0);
            set(pos.x, pos.y, pos.z, d, c);
        }

        __device__
        void set(const uint3 & pos, const float2 & d)
        {
            float3 c=make_float3(0.0,0.0,0.0);
            set(pos.x, pos.y, pos.z, d, c);
        }

        __device__
        void set(const int3 & pos, const float2 &d,const float3 &c)
        {
            set(pos.x, pos.y, pos.z, d, c);
        }

        __device__
        void set(const uint3 & pos, const float2 &d,const float3 &c)
        {
            set(pos.x, pos.y, pos.z, d, c);
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
        
        __device__
        float w_interp(const float3 & pos) const
        {
            const Fptr fp = &Volume::vw;
            return generic_interp(pos,fp);
        }
        
        __device__
        float ww_interp(const float3 & pos) const
        {
            const Fptr fp = &Volume::vww;
            return generic_interp(pos,fp);
        }

        __device__
        float3 rgb_interp(const float3 &p) const
        {
            float3 pos=p;

            float3 rgb;
            const Fptr red_ptr = &Volume::red;
            rgb.x=generic_interp(pos,red_ptr);

            const Fptr green_ptr = &Volume::green;
            rgb.y=generic_interp(pos,green_ptr);

            const Fptr blue_ptr = &Volume::blue;
            rgb.z=generic_interp(pos,blue_ptr);
            return rgb;
        }

        __device__
        float generic_interp(const float3 & pos,const Fptr fp) const;

        __device__ float3 grad(const float3 & pos) const;

        void updateData(const Volume &other)
        {
            size_t s=_resolution.x * _resolution.y * _resolution.z;
            cudaMemcpy(voxels,other.voxels,s*sizeof(tsdfvh::Voxel),cudaMemcpyDeviceToDevice);
        }

        void init(uint3 resolution, float3 dimensions)
        {
//            _resolution = resolution;
//            dim = dimensions;
            
//            uint size=_resolution.x * _resolution.y * _resolution.z;
            
//            cudaMalloc((void**)&voxels, size*sizeof(tsdfvh::Voxel));

//            voxelSize=dim/_resolution;

//            _offset=make_int3(0,0,0);
        }
        
        void initDataFromCpu(VolumeCpu volCpu)
        {
            uint size=_resolution.x * _resolution.y * _resolution.z;            
            cudaMemcpy(voxels, volCpu.voxels, size*sizeof(tsdfvh::Voxel), cudaMemcpyHostToDevice);
        }

        void getCpuData(VolumeCpu &v)
        {
            uint size=_resolution.x * _resolution.y * _resolution.z;
            cudaMemcpy(v.voxels, voxels, size*sizeof(tsdfvh::Voxel), cudaMemcpyDeviceToHost);
        }
        
        __host__ __device__ __forceinline__ 
        bool isPointInside(const float3 &pos) const
        {
            float3 vsize=getSizeInMeters();  
            if( pos.x<0 || pos.x >= vsize.x ||
                pos.y<0 || pos.x >= vsize.y ||
                pos.z<0 || pos.x >= vsize.z)
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
            if(voxels!=nullptr)
                cudaFree(voxels);
        }

    private:

        tsdfvh::Voxel *voxels;
        uint3 _resolution;
        uint3 block_resolution;
        float3 dim;
        float3 voxelSize;
        int3 _offset;

        tsdfvh::HashTable hashTable;

};

//Usefull functions
void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData);
void saveVoxelsToFile(char *fileName, const uint3 &resolution, float vox_size, const tsdfvh::Voxel *voxels);


#include"volume_impl.h"

#endif // VOLUME_H
