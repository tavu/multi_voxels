#ifndef IMAGE_H
#define IMAGE_H
#include <type_traits>
#include <cuda_runtime.h>

#include<iostream>

inline __device__ uint2 thr2pos2()
{
#ifdef __CUDACC__
    return make_uint2( __umul24(blockDim.x, blockIdx.x) + threadIdx.x,
                       __umul24(blockDim.y, blockIdx.y) + threadIdx.y);
#else
    return make_uint2(0,0);
#endif
}

inline __device__ uint thr2pos()
{
#ifdef __CUDACC__
    return threadIdx.x + blockIdx.x*blockDim.x;
#else
    return 0;
#endif
}

struct TrackData
{
    int result;
    float error;
    float J[6];
};



//==========Alocators==========//

struct Ref
{
    Ref(void * d = NULL) :data(d) { }
    void * data;
};

struct Host
{
        Host() :data(NULL) {}

        ~Host() {}

        void alloc(uint size)
        {
            if (data != NULL)
            {
                cudaFreeHost(data);
            }
            cudaHostAlloc(&data, size, cudaHostAllocDefault);
        }

        void release()
        {
            if (data != NULL)
            {
                cudaFreeHost(data);
                data=NULL;
            }
        }
        void * data;
};

struct Device
{
        void * data;
        Device() : data(NULL){}
        ~Device(){}

        void alloc(uint size)
        {
            if (data != NULL)
            {
                cudaFree(data);
            }
            cudaError_t __cudaCalloc_err = cudaMalloc(&data, size);
            if (__cudaCalloc_err == cudaSuccess)
                cudaMemset(data, 0, size);

        }


        void release()
        {
            if (data != NULL)
            {
                cudaFree(data);
                data=NULL;
            }
        }
};

struct HostDevice
{
        HostDevice() :data(NULL) {}
        ~HostDevice() { }

        void alloc(uint size)
        {
            if (data != NULL) {
                cudaFreeHost(data);
            }
            cudaHostAlloc(&data, size, cudaHostAllocMapped);
        }

        void release()
        {
            if (data != NULL)
            {
                cudaFreeHost(data);
                data=NULL;
            }
        }

        void * getDevice() const
        {
            void * devicePtr;
            cudaHostGetDevicePointer(&devicePtr, data, 0);
            return devicePtr;
        }
        void * data;
};
//==========Image==========//

template<typename T, typename Allocator = Ref>
struct Image: public Allocator
{
        typedef T PIXEL_TYPE;
        uint2 size;

        Image() :Allocator()
        {
            size = make_uint2(0, 0);
        }
        Image(const uint2 & s)
        {
            size = make_uint2(0, 0);
            alloc(s);
        }

        void alloc(const uint2 & s)
        {
            if (s.x == size.x && s.y == size.y)
                return;

            Allocator::alloc(s.x * s.y * sizeof(T));
            size = s;
        }

        void copyToHost(void *to) const
        {
            uint s=size.x*size.y*sizeof(T);
            cudaHostAlloc(&to, s, cudaHostAllocDefault);
            cudaMemcpy(to, Allocator::data, s , cudaMemcpyDeviceToHost);
        }

        __device__
        T & el()
        {
            return operator[](thr2pos2());
        }

        __device__
        const T & el() const
        {
            return operator[](thr2pos2());
        }

        __host__ __device__
        T & operator[](const uint2 & pos)
        {
            return static_cast<T *>(Allocator::data)[pos.x + size.x * pos.y];
        }

        __host__ __device__
        const T & operator[](const uint2 & pos) const
        {
            return static_cast<const T *>(Allocator::data)[pos.x + size.x * pos.y];
        }

        Image<T> getDeviceImage()
        {
            return Image<T>(size, Allocator::getDevice());
        }

        operator Image<T>()
        {
            return Image<T>(size, Allocator::data);
        }

        T * data()
        {
            return static_cast<T *>(Allocator::data);
        }

        size_t byteSize() const
        {
            return size.x * size.y * sizeof(T);
        }

        const T * data() const
        {
            return static_cast<const T *>(Allocator::data);
        }
};

template<typename T>
struct Image<T, Ref> : public Ref
{
        typedef T PIXEL_TYPE;
        uint2 size;

        Image()
        {
            size = make_uint2(0, 0);
        }

        Image(const uint2 & s, void * d) :
            Ref(d), size(s)
        {
        }

        __device__
        T & el() {
            return operator[](thr2pos2());
        }

        __device__
        const T & el() const {
            return operator[](thr2pos2());
        }

        __host__ __device__
        T & operator[](const uint2 & pos) {
            return static_cast<T *>(Ref::data)[pos.x + size.x * pos.y];
        }

        __host__ __device__
        const T & operator[](const uint2 & pos) const {
            return static_cast<const T *>(Ref::data)[pos.x + size.x * pos.y];
        }

        T * data() {
            return static_cast<T *>(Ref::data);
        }

        const T * data() const
        {
            return static_cast<const T *>(Ref::data);
        }
};

typedef Image<float, Device> DepthDev;
typedef Image<float, Host> DepthHost;
typedef Image<float3, Host> VertHost;

typedef Image<uchar3, Device> RgbDev;
typedef Image<uchar3, Host> RgbHost;
typedef Image<float3, Host> VertRgb;
#endif
