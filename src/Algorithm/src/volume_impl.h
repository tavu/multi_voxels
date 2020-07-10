#ifndef VOLUME_IMPL_H
#define VOLUME_IMPL_H

__forceinline__ __device__
float Volume::generic_interp(const float3 & pos,const Fptr fp) const
{
    const float3 scaled_pos = make_float3((pos.x * _resolution.x / dim.x) - 0.5f,
                                          (pos.y * _resolution.y / dim.y) - 0.5f,
                                          (pos.z * _resolution.z / dim.z) - 0.5f);
//    const float3 scaled_pos = make_float3( (pos.x /voxelSize.x) - 0.5f ,
//                                          (pos.y /voxelSize.y) - 0.5f ,
//                                          (pos.z /voxelSize.z) - 0.5f );

    const int3 base = make_int3(floorf(scaled_pos));
    const float3 factor = fracf(scaled_pos);
    const int3 lower = max(base, _offset);
    //const int3 upper = min(base + make_int3(1),make_int3(_resolution) - make_int3(1)+_offset);
    const int3 upper = min(base + make_int3(1),maxVoxel() - make_int3(1));

    float tmp0 =( (this->*fp) (make_int3(lower.x, lower.y, lower.z)) * (1 - factor.x) +
                (this->*fp) (make_int3(upper.x, lower.y, lower.z)) * factor.x ) * (1 - factor.y);
    float tmp1 =( (this->*fp) (make_int3(lower.x, upper.y, lower.z)) * (1 - factor.x) +
                (this->*fp) (make_int3(upper.x, upper.y, lower.z)) * factor.x) * factor.y ;
    float tmp2 =( (this->*fp) (make_int3(lower.x, lower.y, upper.z)) * (1 - factor.x) +
                (this->*fp) (make_int3(upper.x, lower.y, upper.z)) * factor.x) * (1 - factor.y);
    float tmp3 =( (this->*fp) (make_int3(lower.x, upper.y, upper.z)) * (1 - factor.x) +
                (this->*fp) (make_int3(upper.x, upper.y, upper.z)) * factor.x) * factor.y;

    return ( (tmp0+tmp1) * (1 - factor.z) + (tmp2+tmp3) * factor.z ) ;
}

__forceinline__ __device__
float3 Volume::grad(const float3 & pos) const
{
    const float3 scaled_pos = make_float3((pos.x * _resolution.x / dim.x) - 0.5f,
                                          (pos.y * _resolution.y / dim.y) - 0.5f,
                                          (pos.z * _resolution.z / dim.z) - 0.5f);
    const int3 base = make_int3(floorf(scaled_pos));
    const float3 factor = fracf(scaled_pos);

    const int3 lower_lower = max(base - make_int3(1), _offset);
    const int3 lower_upper = max(base, _offset);

    const int3 upper_lower = min(base + make_int3(1),
                                 maxVoxel() - make_int3(1));

    const int3 upper_upper = min(base + make_int3(2),
                                 maxVoxel() - make_int3(1));

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
            * make_float3(dim.x / _resolution.x, dim.y / _resolution.y, dim.z / _resolution.z)
            * (0.5f * 0.00003051944088f);
}


#endif
