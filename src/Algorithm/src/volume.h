#ifndef VOLUME_H
#define VOLUME_H

#include"cutil_math.h"
//#include"utils.h"
#include"kparams.h"
#include<iostream>
#include"sMatrix.h"

struct VolumeCpu
{
    uint frame;
    sMatrix4 pose;
    uint3 resolution;
    float3 dimensions;

    short2 *data;
    float3 *color;
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
            data = nullptr;
            color = nullptr;
        }

        bool isNull() const
        {
            return data == nullptr;
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

        __host__ __device__ short2*  getDataPtr() const
        {
            return data;
        }
        
        __host__ __device__ float3*  getColorPtr() const
        {
            return color;
        }

        __host__ __device__ float3 getDimensions() const
        {
            return dim;
        }

        __host__ __device__ __forceinline__ uint getIdx(const uint3 &pos) const
        {
            return pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
        }

        __host__ __device__ __forceinline__ uint getIdx(const int3 &pos) const
        {
            return pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
        }

        __device__
        float2 operator[](const int3 & pos) const
        {
            const short2 d = data[getIdx(pos)];
            return make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f
        }

        __device__
        float2 operator[](const uint3 & pos) const
        {
            uint p=pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
            const short2 d = data[p];
            return make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f
        }

        __device__
        float3 getColor(const int3 & pos) const
        {
            return color[getIdx(pos)];
        }

        __device__
        float3 getColor(const uint3 & pos) const
        {
            uint p=pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
            return color[p];
        }

        __device__
        float vs(const int3 & pos) const
        {
            return data[getIdx(pos)].x;
        }
        
        __device__
        float vw(const int3 & pos) const
        {
            return data[getIdx(pos)].y;
        }
        
        __device__
        float vww(const int3 & pos) const
        {
            short w=data[getIdx(pos)].y;
            if(w>0)
                return 1.0;
            return 0.0;
        }

        __device__
        float red(const int3 & pos) const
        {
            return color[getIdx(pos)].x;
        }

        __device__
        float green(const int3 & pos) const
        {
            return color[getIdx(pos)].y;
        }

        __device__
        float blue(const int3 & pos) const
        {
            //return color[pos.x + pos.y * _size.x + pos.z * _size.x * _size.y].z;
            return color[getIdx(pos)].z;
        }

        __device__
        void set(const int3 & pos, const float2 & d)
        {
            size_t idx=getIdx(pos);
            data[idx] = make_short2(d.x * 32766.0f, d.y);
            color[idx] = make_float3(0.0,0.0,0.0);
        }

        __device__
        void set(const uint3 & pos, const float2 & d)
        {
            uint p=pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
            data[p] = make_short2(d.x * 32766.0f, d.y);
            color[p] = make_float3(0.0,0.0,0.0);
        }

        __device__
        void set(const int3 & pos, const float2 &d,const float3 &c)
        {
            size_t p=getIdx(pos);
            data[p] = make_short2(d.x * 32766.0f, d.y);
            color[p] = c;
        }

        __device__
        void set(const uint3 & pos, const float2 &d,const float3 &c)
        {
            uint p=pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
            data[p] = make_short2(d.x * 32766.0f, d.y);
            color[p] = c;
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
            return generic_interp(pos,fp) * 0.00003051944088f;
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
            cudaMemcpy(data,other.data, s*sizeof(short2),cudaMemcpyDeviceToDevice);
            cudaMemcpy(color,other.color,s*sizeof(float3),cudaMemcpyDeviceToDevice);
        }

        void init(uint3 resolution, float3 dimensions)
        {
            _resolution = resolution;
            dim = dimensions;
            
            uint size=_resolution.x * _resolution.y * _resolution.z;
            
            cudaMalloc((void**)&data, size*sizeof(short2));
            cudaMalloc((void**)&color, size*sizeof(float3));
            
            cudaMemset(data, 0, _resolution.x * _resolution.y * _resolution.z * sizeof(short2));
            cudaMemset(color, 0, _resolution.x * _resolution.y * _resolution.z * sizeof(float3));

            voxelSize=dim/_resolution;

            _offset=make_int3(0,0,0);
        }
        
        void initDataFromCpu(VolumeCpu volCpu)
        {
            uint size=_resolution.x * _resolution.y * _resolution.z;
            cudaMemcpy(data, volCpu.data,size*sizeof(short2),cudaMemcpyHostToDevice);        
            cudaMemcpy(color, volCpu.color,size*sizeof(float3),cudaMemcpyHostToDevice);
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
            if(data!=nullptr)
                cudaFree(data);
            if(color!=nullptr)
                cudaFree(color);

            data=nullptr;
            color=nullptr;
        }

    private:
//        typedef float (Volume::*Fptr)(const uint3&) const;

        uint3 _resolution;
        float3 dim;
        float3 voxelSize;
        int3 _offset;

        short2 *data;
        float3 *color;


};

//Usefull functions
void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData);
void saveVoxelsToFile(char *fileName,const Volume volume);


#include"volume_impl.h"

#endif // VOLUME_H
