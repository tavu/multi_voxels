#include "kfusion.h"
#include <vector_types.h>
#include "constant_parameters.h"
#include "utils.h"
#include "kernels.h"
#include "volume.h"
#include <thrust/device_vector.h>
#include<stdint.h>
#include<iostream>


//static bool firstAcquire = true;
dim3 imageBlock = dim3(32, 16);
dim3 raycastBlock = dim3(32, 8);

KFusion::KFusion(const kparams_t &par, sMatrix4 initPose)
    :params(par),
    _tracked(false)
{
    uint3 vr = make_uint3(params.volume_resolution.x,
                          params.volume_resolution.y,
                          params.volume_resolution.z);

    float3 vd = make_float3(params.volume_size.x,
                            params.volume_size.y,
                            params.volume_size.z);

    volume.init(vr,vd);
    newDataVol.init(vr,vd);

    pose = initPose;
    oldPose=pose;
    this->iterations.clear();
    for(auto it = params.pyramid.begin();it != params.pyramid.end(); it++)
    {    
        this->iterations.push_back(*it);
    }
    largestep=0.75*params.mu;
    inverseCam=getInverseCameraMatrix(params.camera);
    camMatrix=getCameraMatrix(params.camera);
    step = min(params.volume_size) / max(params.volume_resolution);
    viewPose = &pose;

    uint2 cs = make_uint2(params.computationSize.x, params.computationSize.y);
    std::cout<<"CS:"<<cs.x<<","<<cs.y<<std::endl;

    std::cout<<"KAM"<<std::endl;
    std::cout<<camMatrix<<std::endl;

    reduction.alloc(cs);
    vertex.alloc(cs);
    normal.alloc(cs);
    rawDepth.alloc(cs);
    depthImage.alloc(cs);
    rawRgb.alloc(cs);

    scaledDepth.resize(iterations.size());
    inputVertex.resize(iterations.size());
    inputNormal.resize(iterations.size());

    for (int i = 0; i < iterations.size(); ++i)
    {
        scaledDepth[i].alloc(cs >> i);
        inputVertex[i].alloc(cs >> i);
        inputNormal[i].alloc(cs >> i);
    }

    gaussian.alloc(make_uint2(radius * 2 + 1, 1));
    output.alloc(make_uint2(32, 8));
    //generate gaussian array
    generate_gaussian<<< 1,gaussian.size.x>>>(gaussian, delta, radius);
    dim3 grid = divup(dim3(volume.getResolution().x, volume.getResolution().y), imageBlock);

    printCUDAError();
    TICK("initVolume");
    initVolumeKernel<<<grid, imageBlock>>>(volume, make_float2(1.0f, 0.0f));
    TOCK();
    printCUDAError();

    //init new data volume
    initVolumeKernel<<<grid, imageBlock>>>(newDataVol, make_float2(1.0f, 0.0f));
    
    // render buffers
    renderModel.alloc(cs);
    //TODO better memory managment of covariance data
    covData.alloc(cs);
    
    if (printCUDAError())
    {
        cudaDeviceReset();
        exit(1);
    }
}

KFusion::~KFusion()
{
    cudaDeviceSynchronize();
    volume.release();
    
    reduction.release();
    normal.release();
    vertex.release();
    
    for(int i=0;i<inputVertex.size();i++)
    {
        inputVertex[i].release();
    }
    
    for(int i=0;i<inputNormal.size();i++)
    {
        inputNormal[i].release();
    }
     
    for(int i=0;i<scaledDepth.size();i++)
    {
        scaledDepth[i].release();
    }
    
    covData.release();
    rawDepth.release();
    rawRgb.release();
    depthImage.release();
    output.release();
    gaussian.release();
    
    renderModel.release();
    printCUDAError();
}

void KFusion::reset()
{
    dim3 grid = divup(dim3(volume.getResolution().x, volume.getResolution().y), imageBlock);
    initVolumeKernel<<<grid, imageBlock>>>(volume, make_float2(1.0f, 0.0f));
}

void KFusion::updateVolume()
{
    volume.updateData(newDataVol);
}


bool KFusion::preprocessing2(const float *inputDepth,const uchar3 *inputRgb)
{
    cudaMemcpy(rawDepth.data(), inputDepth, params.inputSize.x * params.inputSize.y * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(rawRgb.data(), inputRgb, params.inputSize.x * params.inputSize.y * sizeof(uchar3),cudaMemcpyHostToDevice);

    dim3 grid = divup(make_uint2(params.inputSize.x, params.inputSize.y), imageBlock);
    TICK("bilateral_filter");
    bilateralFilterKernel<<<grid, imageBlock>>>(scaledDepth[0], rawDepth, gaussian, e_delta, radius);
    TOCK();

    return true;
}

bool KFusion::preprocessing(const ushort * inputDepth,const uchar3 *inputRgb)
{
    cudaMemcpy(depthImage.data(), inputDepth, params.inputSize.x * params.inputSize.y * sizeof(ushort), cudaMemcpyHostToDevice);
    TICK("mm2meters");
    mm2metersKernel<<<divup(rawDepth.size, imageBlock), imageBlock>>>(rawDepth, depthImage);
    TOCK();
    cudaMemcpy(rawRgb.data(), inputRgb, params.inputSize.x * params.inputSize.y * sizeof(uchar3),cudaMemcpyHostToDevice);

    // filter the input depth map
    dim3 grid = divup(make_uint2(params.computationSize.x, params.computationSize.y), imageBlock);
    TICK("bilateral_filter");
    bilateralFilterKernel<<<grid, imageBlock>>>(scaledDepth[0], rawDepth, gaussian, e_delta, radius);
    TOCK();        

    return true;
}

bool KFusion::tracking(uint frame)
{
    (void)frame;
    forcePose=false;
    std::vector<dim3> grids;
    for (int i = 0; i < iterations.size(); ++i)
        grids.push_back(divup(make_uint2(params.computationSize.x, params.computationSize.y) >> i, imageBlock));

    // half sample the input depth maps into the pyramid levels
    for (int i = 1; i < iterations.size(); ++i)
    {
        TICK("halfSampleRobust");
        halfSampleRobustImageKernel<<<grids[i], imageBlock>>>(scaledDepth[i], scaledDepth[i-1], e_delta * 3, 1);
        TOCK();
    }

    float4 k = make_float4(params.camera.x, params.camera.y, params.camera.z, params.camera.w);
    // prepare the 3D information from the input depth maps
    for (int i = 0; i < iterations.size(); ++i)
    {
        TICK("depth2vertex");
        depth2vertexKernel<<<grids[i], imageBlock>>>( inputVertex[i], scaledDepth[i], getInverseCameraMatrix(k / float(1 << i))); // inverse camera matrix depends on level
        TOCK();
        TICK("vertex2normal");
        vertex2normalKernel<<<grids[i], imageBlock>>>( inputNormal[i], inputVertex[i] );
        TOCK();
    }

    oldPose = pose;
    const sMatrix4 projectReference = camMatrix*inverse(sMatrix4(&raycastPose));

    for (int level = iterations.size() - 1; level >= 0; --level)
    {
        for (int i = 0; i < iterations[level]; ++i)
        {
            TICK("track");
            trackPose=pose;
            trackKernel<<<grids[level], imageBlock>>>( reduction,
                                                       inputVertex[level],
                                                       inputNormal[level],
                                                       vertex,
                                                       normal,
                                                       sMatrix4( & pose ),
                                                       projectReference,
                                                       dist_threshold,
                                                       normal_threshold);
            TOCK();
            TICK("reduce");
            reduceKernel<<<8, 112>>>( output.data(), reduction, inputVertex[level].size ); // compute the linear system to solve
            TOCK();
            cudaDeviceSynchronize();// important due to async nature of kernel call

            TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(output.data());
            for(int j = 1; j < 8; ++j)
                values[0] += values[j];

            if (updatePoseKernel(pose, output.data(), params.icp_threshold,this->deltaPose))
                break;
        }
    }

    return checkPoseKernel(pose, oldPose, output.data(), params.computationSize,track_threshold);
}

bool KFusion::raycasting(uint frame)
{
    if (frame > 2)
    {
        oldRaycastPose = raycastPose;
        raycastPose = pose;
        dim3 grid=divup(make_uint2(params.computationSize.x,params.computationSize.y),raycastBlock );
        TICK("raycast");
        raycastKernel<<<grid, raycastBlock>>>(vertex, normal, volume, sMatrix4(&raycastPose) * inverseCam,
                                              nearPlane,
                                              farPlane,
                                              step,
                                              largestep,frame);
        TOCK();
    }
    else
    {
        return false;
    }

    printCUDAError();

    return true;
}

void KFusion::integrateNewData(sMatrix4 p)
{
    dim3 grid=divup(dim3(newDataVol.getResolution().x, newDataVol.getResolution().y), imageBlock);
    //initVolumeKernel<<<grid, imageBlock>>>(newDataVol, make_float2(1.0f, 0.0f));


    integrateKernel<<<grid,imageBlock>>>(newDataVol,rawDepth,rawRgb,
                                         inverse(p),camMatrix,params.mu,maxweight );

}

void KFusion::integrateSlices(VolumeSlices &slices)
{

}



bool KFusion::integration(uint frame)
{
    //bool doIntegrate = checkPoseKernel(pose, oldPose, output.data(),params.computationSize, track_threshold);
    if (_tracked || frame <= 3)
    {
        printCUDAError();
        TICK("integrate");
        dim3 grid=divup(dim3(volume.getResolution().x, volume.getResolution().y), imageBlock);
        integrateKernel<<<grid, imageBlock>>>(volume,
                                              rawDepth,
                                              rawRgb,
                                              inverse(pose),
                                              camMatrix,
                                              params.mu,
                                              maxweight );

        TOCK();       
        return true;
    }

    return false;
}

bool KFusion::deIntegration(sMatrix4 p,const Host &depth,const Host &rgb)
{
    image_copy(rawDepth,depth, rawDepth.size.x*rawDepth.size.y*sizeof(float));
    image_copy(rawRgb,rgb, rawRgb.size.x*rawRgb.size.y*sizeof(uchar3));

    TICK("deintegrate");
    deIntegrateKernel<<<divup(dim3(volume.getResolution().x, volume.getResolution().y), imageBlock), imageBlock>>>(volume,
                                                                                           rawDepth,
                                                                                           rawRgb,
                                                                                           inverse(sMatrix4(&p)),
                                                                                           camMatrix,
                                                                                           params.mu,
                                                                                           maxweight);    
    TOCK();
    return true;
}

bool KFusion::reIntegration(sMatrix4 p,const Host &depth,const Host &rgb)
{    
    uint s = params.inputSize.x*params.inputSize.y;
    image_copy(rawDepth,depth, s*sizeof(float));
    image_copy(rawRgb,rgb, s*sizeof(uchar3));
    TICK("reintegrate");
    integrateKernel<<<divup(dim3(volume.getResolution().x, volume.getResolution().y), imageBlock), imageBlock>>>(volume,
                                                                                           rawDepth,
                                                                                           rawRgb,
                                                                                           inverse(sMatrix4(&p)),
                                                                                           camMatrix,
                                                                                           params.mu,
                                                                                           maxweight );
    TOCK();
    return true;
}

Image<float3, Host> KFusion::getAllVertex()
{
    Image<float3, Host> ret( make_uint2(params.inputSize.x, params.inputSize.y) );
    cudaMemcpy(ret.data(), inputVertex[0].data(),
            params.inputSize.x * params.inputSize.y * sizeof(float3),
            cudaMemcpyDeviceToHost);
    return ret;
}

Image<float3, Host> KFusion::getAllNormals()
{
    Image<float3, Host> ret( make_uint2(params.inputSize.x, params.inputSize.y) );
    cudaMemcpy(ret.data(), inputNormal[0].data(),
            params.inputSize.x * params.inputSize.y * sizeof(float3),
            cudaMemcpyDeviceToHost);
    return ret;
}

Image<TrackData, Host> KFusion::getTrackData()
{
    Image<TrackData, Host> trackData;
    trackData.alloc(reduction.size);

    cudaMemcpy(trackData.data(), reduction.data(),reduction.size.x*reduction.size.y*sizeof(TrackData),cudaMemcpyDeviceToHost);

    return trackData;
}


void KFusion::getVertices(std::vector<float3> &vertices)
{
    vertices.clear();
    short2 *hostData = (short2 *) malloc(volume.getResolution().x * volume.getResolution().y * volume.getResolution().z * sizeof(short2));

    if (cudaMemcpy(hostData,
                   volume.getDataPtr(),
                   volume.getResolution().x *
                   volume.getResolution().y *
                   volume.getResolution().z *
                   sizeof(short2),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cerr << "Error reading volumetric representation data from the GPU. "<< std::endl;
        exit(1);
    }
    generateTriangles(vertices, volume, hostData);
    free(hostData);
}

void KFusion::renderVolume(uchar3 * out)
{
    dim3 grid=divup(renderModel.size,imageBlock);

    TICK("renderVolume");
    renderVolumeKernel2<<<grid,imageBlock>>>( renderModel,vertex,normal,light,ambient,nearPlane,farPlane);
    TOCK();
    

    cudaMemcpy(out, renderModel.data(),
            params.computationSize.x * params.computationSize.y * sizeof(uchar3),
            cudaMemcpyDeviceToHost);
}

Image<float, Host> KFusion::vertex2Depth()
{
    Image<float, Host> ret(params.inputSize);
//    Image<float, Device> model(params.inputSize);
    
//     dim3 grid=divup(model.size,imageBlock);
//    vertex2depthKernel<<<grid,imageBlock>>>( model,vertex,normal,nearPlane,farPlane,K);
    
//    cudaMemcpy(ret.data(), model.data(),
//            params.inputSize.x * params.inputSize.y * sizeof(float),
//            cudaMemcpyDeviceToHost);
    return ret;
}

float KFusion::compareRgb( )
{
    Image<float, Device> diff( make_uint2(params.inputSize.x, params.inputSize.y) );
    compareRgbKernel<<<divup(renderModel.size, imageBlock), imageBlock>>>( renderModel,rawRgb,diff);
    
    size_t size=params.inputSize.x*params.inputSize.y;
    thrust::device_ptr<float> diff_ptr(diff.data());
    thrust::device_vector<float> d_vec(diff_ptr,diff_ptr+size);
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<float>());

    float ret = sum/size;
    
    diff.release();
    return ret;
}

void KFusion::getImageProjection(sMatrix4 p, uchar3 *out)
{
    Image<float3, Device> vertexNew, normalNew;
    vertexNew.alloc(params.inputSize);
    normalNew.alloc(params.inputSize);

    dim3 grid=divup(params.inputSize,raycastBlock );
    //raycast from given pose
    printCUDAError();
    raycastKernel<<<grid, raycastBlock>>>(vertexNew, normalNew, volume, p * inverseCam,
                                         nearPlane,farPlane,step,largestep,1);
    
    cudaDeviceSynchronize();
    printCUDAError();

    grid=divup(params.inputSize,imageBlock );
    renderRgbKernel<<<grid, imageBlock>>>( renderModel,volume,vertexNew,normalNew);

    cudaMemcpy(out, renderModel.data(),
               params.inputSize.x * params.inputSize.y * sizeof(uchar3),
               cudaMemcpyDeviceToHost);
    
    vertexNew.release();
    normalNew.release();
}

float KFusion::getWrongNormalsSize()
{

    dim3 grid=divup(make_uint2(params.computationSize.x,params.computationSize.y),raycastBlock );

    Image<int, Device> model;
    model.alloc(params.inputSize);

    wrongNormalsSizeKernel<<<grid, raycastBlock>>>( model,reduction );

    size_t size=params.inputSize.x*params.inputSize.y;

    thrust::device_ptr<int> diff_ptr(model.data());
    thrust::device_vector<int> d_vec(diff_ptr,diff_ptr+size);
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

    float ret = (float)sum/(params.computationSize.x*params.computationSize.y);
    
    model.release();
    return ret;
}

void KFusion::renderImage(uchar3 * out)
{
    TICK("renderVolume");
    cudaDeviceSynchronize();
    dim3 grid=divup(renderModel.size, imageBlock);
    renderRgbKernel<<<grid, imageBlock>>>(renderModel,volume,vertex,normal);
    TOCK();

     cudaMemcpy(out, renderModel.data(),
                params.computationSize.x * params.computationSize.y * sizeof(uchar3),
                cudaMemcpyDeviceToHost);

}

void KFusion::renderTrack(uchar3 * out)
{
    dim3 grid=divup(renderModel.size, imageBlock);
    TICK("renderTrack");
    renderTrackKernel<<<grid, imageBlock>>>( renderModel, reduction );
    TOCK();
    cudaMemcpy(out, renderModel.data(), params.inputSize.x * params.inputSize.y * sizeof(uchar3), cudaMemcpyDeviceToHost);

    printCUDAError();
}

void KFusion::renderDepthFromVertex(uchar3 * out)
{
    Image<float, Device> depth;
    depth.alloc(rawDepth.size);
    dim3 grid=divup(renderModel.size, imageBlock);

    vertex2depthKernel<<<grid, imageBlock>>>( depth, inputVertex[0], camMatrix);
    renderDepthKernel<<<grid, imageBlock>>>( renderModel, depth, nearPlane, farPlane );
    cudaMemcpy(out,renderModel.data(), params.inputSize.x * params.inputSize.y * sizeof(uchar3), cudaMemcpyDeviceToHost);
}

void KFusion::renderDepth(uchar3 * out)
{
    TICK("renderDepthKernel");
    dim3 grid=divup(renderModel.size, imageBlock);
    renderDepthKernel<<<grid, imageBlock>>>( renderModel, rawDepth, nearPlane, farPlane );
    TOCK();
    cudaMemcpy(out,renderModel.data(), params.inputSize.x * params.inputSize.y * sizeof(uchar3), cudaMemcpyDeviceToHost);
}

bool KFusion::updatePoseKernel(sMatrix4 & pose, const float * output,float icp_threshold,sMatrix4 &deltaPose)
{
    // Update the pose regarding the tracking result
    TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
    TooN::Vector<6> x = solve(values[0].slice<1, 27>());
    TooN::SE3<> delta(x);
    sMatrix4 deltaMat=tosMatrix4(delta);
    sMatrix4 delta4 = deltaMat * sMatrix4(&pose);

    pose.data[0].x = delta4.data[0].x;
    pose.data[0].y = delta4.data[0].y;
    pose.data[0].z = delta4.data[0].z;
    pose.data[0].w = delta4.data[0].w;
    pose.data[1].x = delta4.data[1].x;
    pose.data[1].y = delta4.data[1].y;
    pose.data[1].z = delta4.data[1].z;
    pose.data[1].w = delta4.data[1].w;
    pose.data[2].x = delta4.data[2].x;
    pose.data[2].y = delta4.data[2].y;
    pose.data[2].z = delta4.data[2].z;
    pose.data[2].w = delta4.data[2].w;
    pose.data[3].x = delta4.data[3].x;
    pose.data[3].y = delta4.data[3].y;
    pose.data[3].z = delta4.data[3].z;
    pose.data[3].w = delta4.data[3].w;

    // Return validity test result of the tracking
    if (norm(x) < icp_threshold)
    {
        deltaPose=deltaMat;
        return true;
    }
    return false;
}

bool KFusion::checkPoseKernel(sMatrix4 & pose,
                     sMatrix4 oldPose,
                     const float * output,
                     uint2 imageSize,
                     float track_threshold)
{
    if(forcePose)
    {
        _tracked=true;
        return true;
    }
    
    // Check the tracking result, and go back to the previous camera position if necessary
    // return true;
    TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

    if ( (std::sqrt(values(0, 0) / values(0, 28)) > 2e-2) ||
         (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold) )
    {
        pose = oldPose;
        _tracked=false;
        return false;
    }

    _tracked=true;
    poseInv=inverse(pose);
    return true;
}

void KFusion::getImageRaw(RgbHost &to) const
{

  uint s=(uint)rawRgb.size.x*rawRgb.size.y*sizeof(uchar3);
  to.alloc(rawRgb.size);
  cudaMemcpy(to.data(), rawRgb.data(),s,cudaMemcpyDeviceToHost);
//  to.size=rawRgb.size;
}

void KFusion::getDepthRaw(DepthHost &to) const
{
    uint s=(uint)rawDepth.size.x*rawDepth.size.y*sizeof(float);
    to.alloc(rawDepth.size);
    cudaMemcpy(to.data(), rawDepth.data(),s,cudaMemcpyDeviceToHost);
    to.size=rawDepth.size;
}

void KFusion::getIcpValues(Image<float3, Host> &depthVertex,
                             Image<float3, Host> &raycastVertex,
                             Image<float3, Host> &raycastNormals,
                             Image<TrackData, Host> &trackData) const
{
    uint s=(uint) (params.volume_size.x*params.volume_size.y);
    depthVertex.alloc(inputVertex[0].size);
    raycastVertex.alloc(vertex.size);
    raycastNormals.alloc(normal.size);
    trackData.alloc(reduction.size);
  
    cudaMemcpy(depthVertex.data(), inputVertex[0].data(),s*sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(raycastVertex.data(), vertex.data(),s*sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(raycastNormals.data(), normal.data(),s*sizeof(float3),cudaMemcpyDeviceToHost);
    cudaMemcpy(trackData.data(), reduction.data(),reduction.size.x*reduction.size.y*sizeof(TrackData),cudaMemcpyDeviceToHost);
}

float KFusion::getFitness()
{
    size_t size=reduction.size.x*reduction.size.y;
    thrust::device_ptr<TrackData> ptr=thrust::device_pointer_cast(reduction.data());
    TrackData d;
    d.result=1;
    int count = thrust::count(ptr,ptr+size,d);

    float ret=(float)count/size;
    return ret;
}


sMatrix6 KFusion::calculate_ICP_COV()
{
    sMatrix4 currPose=pose;
    sMatrix4 invPrevPose=inverse(oldPose);
    sMatrix4 delta=invPrevPose*currPose;

    sMatrix4 projectedReference = camMatrix*inverse(sMatrix4(&raycastPose));
    dim3 grid=divup(make_uint2(params.inputSize.x,params.inputSize.y),imageBlock );

    sMatrix6 initMat;
    for(int i=0;i<36;i++)
        initMat.data[i]=0.0;



    icpCovarianceFirstTerm<<<grid, imageBlock>>>(inputVertex[0],
                                                vertex,
                                                normal,
                                                reduction,
                                                covData,
                                                trackPose,
                                                projectedReference,
                                                delta,
                                                params.cov_big);
    
    cudaDeviceSynchronize();    
    size_t size=covData.size.x*covData.size.y;
    thrust::device_ptr<sMatrix6> cov_ptr(covData.data());
    sMatrix6 d2J_dX2 = thrust::reduce(cov_ptr, cov_ptr+size, initMat, thrust::plus<sMatrix6>());

    icpCovarianceSecondTerm<<<grid, imageBlock>>>(inputVertex[0],
                                                  vertex,
                                                  normal,
                                                  reduction,
                                                  covData,
                                                  trackPose,
                                                  projectedReference,
                                                  delta,                                                  
                                                  params.cov_z,
                                                  params.cov_big);
    cudaDeviceSynchronize();
    sMatrix6 covSecondTerm = thrust::reduce(cov_ptr, cov_ptr+size, initMat, thrust::plus<sMatrix6>());


    sMatrix6 d2J_dX2inv=inverse(d2J_dX2);
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

    return ret;
}



