#include"kernelscalls.h"
#include"kernels.h"
#include <thrust/device_vector.h>
#include "constant_parameters.h"
#include<iostream>


sMatrix6 calculatePoint2PointCov(const std::vector<float3> &vert,
                                 int vertSize,
                                 const std::vector<float3> &prevVert,
                                 int prevVertSize,
                                 //const int2 *corresp,
                                 const std::vector<int> &sourceCorr,
                                 const std::vector<int> &targetCorr,
                                 const sMatrix4 &tf,
                                 const kparams_t &params)
{
    float3 *vertGpu;
    cudaMalloc(&vertGpu,vertSize*sizeof(float3));
    cudaMemcpy(vertGpu,vert.data(),vertSize*sizeof(float3),cudaMemcpyHostToDevice);

    float3 *prevVertGpu;
    cudaMalloc(&prevVertGpu,prevVertSize*sizeof(float3));
    cudaMemcpy(prevVertGpu,prevVert.data(),prevVertSize*sizeof(float3),cudaMemcpyHostToDevice);

    size_t correspSize=sourceCorr.size();
    int *sourceCorrGpu;
    int err=cudaMalloc(&sourceCorrGpu,correspSize*sizeof(int));
    cudaMemcpy(sourceCorrGpu,sourceCorr.data(),correspSize*sizeof(int),cudaMemcpyHostToDevice);

    int *targetCorrGpu;
    err=cudaMalloc(&targetCorrGpu,correspSize*sizeof(int));
    cudaMemcpy(targetCorrGpu,targetCorr.data(),correspSize*sizeof(int),cudaMemcpyHostToDevice);
    
    sMatrix6 *covData;
    cudaMalloc(&covData,correspSize*sizeof(sMatrix6));
    
    point2PointCovFirstTerm<<<(correspSize+256)/256, 256>>>(vertGpu,
                                                            vertSize,
                                                            prevVertGpu,
                                                            prevVertSize,
                                                            sourceCorrGpu,
                                                            targetCorrGpu,
                                                            correspSize,
                                                            tf,
                                                            covData,
                                                            params.cov_big);
    sMatrix6 initMat;
    for(int i=0;i<36;i++)
        initMat.data[i]=0.0;
    
    cudaDeviceSynchronize();
    
    
    
    thrust::device_ptr<sMatrix6> cov_ptr(covData);
    sMatrix6 d2J_dX2 = thrust::reduce(cov_ptr, cov_ptr+correspSize, initMat, thrust::plus<sMatrix6>());

    //float cov_z=0.05;
    float cov_z=1;
    point2PointCovSecondTerm<<<(correspSize+256)/256, 256>>>(vertGpu,
                                                            vertSize,
                                                            prevVertGpu,
                                                            prevVertSize,
                                                            sourceCorrGpu,
                                                            targetCorrGpu,
                                                            correspSize,
                                                            tf,
                                                            cov_z,
                                                            covData,
                                                            params.cov_big);
    cudaDeviceSynchronize();    
    sMatrix6 covSecondTerm = thrust::reduce(cov_ptr, cov_ptr+correspSize, initMat, thrust::plus<sMatrix6>());

//    std::cout<<"d2J_dX2"<<std::endl;
//    std::cout<<d2J_dX2<<std::endl;

    sMatrix6 d2J_dX2inv=inverse(d2J_dX2);


//    std::cout<<"d2J_dX2inv"<<std::endl;
//    std::cout<<d2J_dX2inv<<std::endl;

    sMatrix6 tmp=d2J_dX2inv * covSecondTerm;
    sMatrix6 icpCov= tmp * d2J_dX2inv;
    
    //make sure that covariance matrix is symetric.
    //small asymetries may occur due to numerical stability
    sMatrix6 ret;
    for(int i=0;i<6;i++)
    {
        for(int j=0;j<6;j++)
        {
            //eliminate NaN values
            if(icpCov(i,j)!=icpCov(i,j))
            {
                icpCov(i,j)=params.cov_big;
            }
            if(icpCov(j,i)!=icpCov(j,i))
            {
                icpCov(j,i)=params.cov_big;
            }
            float val=( icpCov(i,j) + icpCov(j,i))/2;
            ret(i,j)=val;
            ret(j,i)=val;

        }
    }
    
    cudaFree(vertGpu);
    cudaFree(prevVertGpu);
    cudaFree(sourceCorrGpu);
    cudaFree(targetCorrGpu);
    cudaFree(covData);
    
    return ret;
}
