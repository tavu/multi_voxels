#ifndef DEVICE_CODE_H
#define DEVICE_CODE_H

#include"volume.h"
__forceinline__ __host__  __device__  void eulerFromHomo(const sMatrix4 &pose,float &roll,float &pitch,float &yaw)
{
    roll  = atan2f(pose(2,1), pose(2,2));
    pitch = asinf(-pose(2,0));
    yaw   = atan2f(pose(1,0), pose(0,0));
}

__forceinline__ __host__ __device__ int  signf(const float a)
{
    return a>0?1:-1;
}


__forceinline__ __host__ __device__ int  sign(const int a)
{
    return a>0?1:-1;
}

__forceinline__ __host__ __device__ void  swapf(float &f1,float &f2)
{
    float tmp=f2;
    f2=f1;
    f1=tmp;
}


__forceinline__ __host__ __device__ float sq(const float x)
{
    return x * x;
}


__device__ __forceinline__ float4 raycast(const Volume volume,
                                          const uint2 pos,
                                          const sMatrix4 view,
                                          const float nearPlane,
                                          const float farPlane,
                                          const float step,
                                          const float largestep,int frame)
{
    const float3 origin = view.get_translation();
    const float3 direction = rotate(view, make_float3(pos.x, pos.y, 1.f));

    const float3 invR = make_float3(1.0f) / direction;
    const float3 tbot = -1 * invR * origin;
    const float3 ttop = invR * (volume.getDimensions() - origin);

    // re-order intersections to find smallest and largest on each axis
    const float3 tmin = fminf(ttop, tbot);
    const float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y),
                                     fmaxf(tmin.x, tmin.z));
    const float smallest_tmax = fminf(fminf(tmax.x, tmax.y),
                                      fminf(tmax.x, tmax.z));

    // check against near and far plane
    const float tnear = fmaxf(largest_tmin, nearPlane);
    const float tfar = fminf(smallest_tmax, farPlane);

    if (tnear < tfar)
    {
        // first walk with largesteps until we found a hit
        float t = tnear;
        float stepsize = largestep;
        float f_t = volume.interp(origin + direction * t);
        float f_tt = 0;

        if (f_t > 0) // ups, if we were already in it, then don't render anything here
        {
            for (; t < tfar; t += stepsize)
            {
                f_tt = volume.interp(origin + direction * t);
                if (f_tt < 0)                  // got it, jump out of inner loop
                    break;
                if (f_tt < 0.8f)               // coming closer, reduce stepsize
                    stepsize = step;
                f_t = f_tt;
            }
            if (f_tt < 0)
            {           // got it, calculate accurate intersection
                t = t + stepsize * f_tt / (f_t - f_tt);
                return make_float4(origin + direction * t, t);
            }
        }
    }
    return make_float4(0);
}

#endif // DEVICE_CODE_H
