#ifndef KERNELS_CALLS_H
#define KERNELS_CALLS_H

#include<utils.h>
#include"kparams.h"
#include<vector>
sMatrix6 calculatePoint2PointCov(const std::vector<float3> &vert,
                                 int vertSize,
                                 const std::vector<float3> &prevVert,
                                 int prevVertSize,
                                 const std::vector<int> &sourceCorr,
                                 const std::vector<int> &targetCorr,
                                 const sMatrix4 &tf,
                                 const kparams_t &params);
#endif
