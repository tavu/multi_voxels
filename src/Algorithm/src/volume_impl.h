#ifndef VOLUME_IMPL_H
#define VOLUME_IMPL_H

#include"tsdfvh/voxel.h"

__forceinline__ __device__
tsdfvh::Voxel Volume::getVoxelInterp(const float3 &pos,int &blockIdx,bool useColor) const
{
    tsdfvh::Voxel ret;
    const float3 scaled_pos = make_float3((pos.x * getResolution().x / getDimensions().x) - 0.5f,
                                          (pos.y * getResolution().y / getDimensions().y) - 0.5f,
                                          (pos.z * getResolution().z / getDimensions().z) - 0.5f);

    const int3 max_voxel=make_int3(getResolution().x - 1,
                                   getResolution().y - 1,
                                   getResolution().z - 1);

    const int3 base = make_int3(floorf(scaled_pos));
    const float3 factor = fracf(scaled_pos);
    const int3 lower = max(base, make_int3(0));
    const int3 upper = min(base + make_int3(1),max_voxel);

    voxel_t *v[8];

    v[0]=getVoxel(make_int3(lower.x, lower.y, lower.z),blockIdx);
    v[1]=getVoxel(make_int3(upper.x, lower.y, lower.z),blockIdx);
    v[2]=getVoxel(make_int3(lower.x, upper.y, lower.z),blockIdx);
    v[3]=getVoxel(make_int3(upper.x, upper.y, lower.z),blockIdx);
    v[4]=getVoxel(make_int3(lower.x, lower.y, upper.z),blockIdx);
    v[5]=getVoxel(make_int3(upper.x, lower.y, upper.z),blockIdx);
    v[6]=getVoxel(make_int3(lower.x, upper.y, upper.z),blockIdx);
    v[7]=getVoxel(make_int3(upper.x, upper.y, upper.z),blockIdx);


    float sdf[8];
    float3 col[8];
    for(int i=0;i<8;i++)
    {
        if(v[i]==nullptr)
        {
            col[i]=make_float3(0.0, 0.0, 0.0);
            sdf[i]=1.0;
        }
        else
        {
            sdf[i]=v[i]->getTsdf();
            col[i]=v[i]->color;
        }
    }

    float tmp0 = (sdf[0] * (1 - factor.x) +
                  sdf[1] * factor.x ) * (1 - factor.y);
    float tmp1 = (sdf[2] * (1 - factor.x) +
                  sdf[3] * factor.x) * factor.y ;
    float tmp2 = (sdf[4] * (1 - factor.x) +
                  sdf[5] * factor.x) * (1 - factor.y);
    float tmp3 = (sdf[6] * (1 - factor.x) +
                  sdf[7] * factor.x) * factor.y;

    ret.setTsdf( (tmp0+tmp1) * (1 - factor.z) + (tmp2+tmp3) * factor.z );

    if(useColor)
    {
        float r0 = (col[0].x * (1 - factor.x) +
                    col[1].x * factor.x ) * (1 - factor.y);
        float r1 = (col[2].x * (1 - factor.x) +
                    col[3].x * factor.x) * factor.y ;
        float r2 = (col[4].x * (1 - factor.x) +
                    col[5].x * factor.x) * (1 - factor.y);
        float r3 = (col[6].x * (1 - factor.x) +
                    col[7].x * factor.x) * factor.y;
        r0=( (r0+r1) * (1 - factor.z) + (r2+r3) * factor.z ) ;


        float g0 = (col[0].y * (1 - factor.x) +
                    col[1].y * factor.x ) * (1 - factor.y);
        float g1 = (col[2].y * (1 - factor.x) +
                    col[3].y * factor.x) * factor.y ;
        float g2 = (col[4].y * (1 - factor.x) +
                    col[5].y * factor.x) * (1 - factor.y);
        float g3 = (col[6].y * (1 - factor.x) +
                    col[7].y * factor.x) * factor.y;
        g0=( (g0+g1) * (1 - factor.z) + (g2+g3) * factor.z ) ;


        float b0 = (col[0].z * (1 - factor.x) +
                    col[1].z * factor.x ) * (1 - factor.y);
        float b1 = (col[2].z * (1 - factor.x) +
                    col[3].z * factor.x) * factor.y ;
        float b2 = (col[4].z * (1 - factor.x) +
                    col[5].z * factor.x) * (1 - factor.y);
        float b3 = (col[6].z * (1 - factor.x) +
                    col[7].z * factor.x) * factor.y;

        b0=( (b0+b1) * (1 - factor.z) + (b2+b3) * factor.z ) ;

        ret.setColor(make_float3(r0,g0,b0));
    }

    return ret;
}


__forceinline__ __device__
float3 Volume::grad(const float3 & pos) const
{
    const float3 scaled_pos = make_float3((pos.x * getResolution().x / getDimensions().x) - 0.5f,
                                          (pos.y * getResolution().y / getDimensions().y) - 0.5f,
                                          (pos.z * getResolution().z / getDimensions().z) - 0.5f);
    const int3 base = make_int3(floorf(scaled_pos));
    const float3 factor = fracf(scaled_pos);

    const int3 max_voxel=make_int3(getResolution().x - 1,
                                   getResolution().y - 1,
                                   getResolution().z - 1);

    const int3 lower_lower = max(base - make_int3(1), make_int3(0));
    const int3 lower_upper = max(base, make_int3(0));

    const int3 upper_lower = min(base + make_int3(1),
                                 max_voxel);

    const int3 upper_upper = min(base + make_int3(2),
                                 max_voxel);

    const int3 & lower = lower_upper;
    const int3 & upper = upper_lower;

    float3 gradient;

    gradient.x = ((
        ( vs(make_int3(upper_lower.x, lower.y, lower.z))-vs(make_int3(lower_lower.x, lower.y, lower.z))) * (1 - factor.x)
        + ( vs(make_int3(upper_upper.x, lower.y, lower.z))-vs(make_int3(lower_upper.x, lower.y, lower.z))) * factor.x) * (1 - factor.y)
        + ( (vs(make_int3(upper_lower.x, upper.y, lower.z)) - vs(make_int3(lower_lower.x, upper.y, lower.z)))* (1 - factor.x)
            + (vs(make_int3(upper_upper.x, upper.y, lower.z))- vs(make_int3(lower_upper.x, upper.y,lower.z))) * factor.x) * factor.y) * (1 - factor.z)
                 + (((vs(make_int3(upper_lower.x, lower.y, upper.z))
                      - vs(make_int3(lower_lower.x, lower.y, upper.z)))
                     * (1 - factor.x)
                     + (vs(make_int3(upper_upper.x, lower.y, upper.z))
                        - vs(
                            make_int3(lower_upper.x, lower.y,
                                       upper.z))) * factor.x)
                    * (1 - factor.y)
                    + ((vs(make_int3(upper_lower.x, upper.y, upper.z))
                        - vs(
                            make_int3(lower_lower.x, upper.y,
                                       upper.z))) * (1 - factor.x)
                       + (vs(
                              make_int3(upper_upper.x, upper.y,
                                         upper.z))
                          - vs(
                              make_int3(lower_upper.x,
                                         upper.y, upper.z)))
                       * factor.x) * factor.y) * factor.z;

    gradient.y =
            (((vs(make_int3(lower.x, upper_lower.y, lower.z))
               - vs(make_int3(lower.x, lower_lower.y, lower.z)))
              * (1 - factor.x)
              + (vs(make_int3(upper.x, upper_lower.y, lower.z))
                 - vs(
                     make_int3(upper.x, lower_lower.y,
                                lower.z))) * factor.x)
             * (1 - factor.y)
             + ((vs(make_int3(lower.x, upper_upper.y, lower.z))
                 - vs(
                     make_int3(lower.x, lower_upper.y,
                                lower.z))) * (1 - factor.x)
                + (vs(
                       make_int3(upper.x, upper_upper.y,
                                  lower.z))
                   - vs(
                       make_int3(upper.x,
                                  lower_upper.y, lower.z)))
                * factor.x) * factor.y) * (1 - factor.z)
            + (((vs(make_int3(lower.x, upper_lower.y, upper.z))
                 - vs(
                     make_int3(lower.x, lower_lower.y,
                                upper.z))) * (1 - factor.x)
                + (vs(
                       make_int3(upper.x, upper_lower.y,
                                  upper.z))
                   - vs(
                       make_int3(upper.x,
                                  lower_lower.y, upper.z)))
                * factor.x) * (1 - factor.y)
               + ((vs(
                       make_int3(lower.x, upper_upper.y,
                                  upper.z))
                   - vs(
                       make_int3(lower.x,
                                  lower_upper.y, upper.z)))
                  * (1 - factor.x)
                  + (vs(
                         make_int3(upper.x,
                                    upper_upper.y, upper.z))
                     - vs(
                         make_int3(upper.x,
                                    lower_upper.y,
                                    upper.z)))
                  * factor.x) * factor.y)
            * factor.z;

    gradient.z = (((vs(make_int3(lower.x, lower.y, upper_lower.z))
                    - vs(make_int3(lower.x, lower.y, lower_lower.z)))
                   * (1 - factor.x)
                   + (vs(make_int3(upper.x, lower.y, upper_lower.z))
                      - vs(make_int3(upper.x, lower.y, lower_lower.z)))
                   * factor.x) * (1 - factor.y)
                  + ((vs(make_int3(lower.x, upper.y, upper_lower.z))
                      - vs(make_int3(lower.x, upper.y, lower_lower.z)))
                     * (1 - factor.x)
                     + (vs(make_int3(upper.x, upper.y, upper_lower.z))
                        - vs(
                            make_int3(upper.x, upper.y,
                                       lower_lower.z))) * factor.x)
                  * factor.y) * (1 - factor.z)
                 + (((vs(make_int3(lower.x, lower.y, upper_upper.z))
                      - vs(make_int3(lower.x, lower.y, lower_upper.z)))
                     * (1 - factor.x)
                     + (vs(make_int3(upper.x, lower.y, upper_upper.z))
                        - vs(
                            make_int3(upper.x, lower.y,
                                       lower_upper.z))) * factor.x)
                    * (1 - factor.y)
                    + ((vs(make_int3(lower.x, upper.y, upper_upper.z))
                        - vs(
                            make_int3(lower.x, upper.y,
                                       lower_upper.z)))
                       * (1 - factor.x)
                       + (vs(
                              make_int3(upper.x, upper.y,
                                         upper_upper.z))
                          - vs(
                              make_int3(upper.x, upper.y,
                                         lower_upper.z)))
                       * factor.x) * factor.y) * factor.z;

    return gradient
            * make_float3(voxelSize.x, voxelSize.y, voxelSize.z)
            * (0.5f * 0.00003051944088f);
}


#endif
