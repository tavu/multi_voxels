#ifndef VOLUME_H
#define VOLUME_H

#include"cutil_math.h"
#include"kparams.h"
#include<iostream>
#include"sMatrix.h"

#include"tsdfvh/voxel.h"

//for short x
//x * 0.00003051944088f
//float2 ret = make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f
//data[p] = make_short2(d.x * 32766.0f, d.y);
//float2 ret = make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f

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

    public:
        Volume()
        {
            _resolution = make_uint3(0);
            dim = make_float3(1);
            voxels = nullptr;
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
            uint idx=getIdx(x,y,z);
            return voxels[idx];
        }

        __device__ __forceinline__
        void setVoxel(const tsdfvh::Voxel &v, int x, int y, int z)
        {
            uint idx=getIdx(x,y,z);
            voxels[idx]=v;
        }

        //IDX
        __host__ __device__ __forceinline__
        uint getIdx(int x, int y, int z) const
        {
            return x + y * _resolution.x + z * _resolution.x * _resolution.y;
        }

        __host__ __device__ __forceinline__
        uint getIdx(const uint3 &pos) const
        {
            return getIdx(pos.x, pos.y, pos.z);
        }

        __host__ __device__ __forceinline__
        uint getIdx(const int3 &pos) const
        {
            return getIdx(pos.x, pos.y, pos.z);
        }


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
            uint idx=getIdx(x, y, z);
            //return color[idx];
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
//            uint idx=getIdx(x,y,z);

//            voxels[idx].color=c;
//            voxels[idx].sdf=d.x;
//            voxels[idx].weight=d.y;
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
            _resolution = resolution;
            dim = dimensions;
            
            uint size=_resolution.x * _resolution.y * _resolution.z;
            
            cudaMalloc((void**)&voxels, size*sizeof(tsdfvh::Voxel));

            voxelSize=dim/_resolution;

            _offset=make_int3(0,0,0);
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
        float3 dim;
        float3 voxelSize;
        int3 _offset;

};

//Usefull functions
void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData);
void saveVoxelsToFile(char *fileName,const Volume volume);


#include"volume_impl.h"

#endif // VOLUME_H
