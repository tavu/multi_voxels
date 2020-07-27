#include "fusionHandler.h"
#include<iostream>
#include"utils.h"

#include"defs.h"
#include"constant_parameters.h"
#include <unistd.h>
#include"volume.h"

#include"kfusion.h"
#include"tsdfvh/voxel.h"

#include<fstream>

FusionHandler::FusionHandler(const kparams_t &p,sMatrix4 initPose)
    :params(p)
{
    _fusion = new KFusion(params,initPose);
}

bool FusionHandler::processFrame(int _frame, const float *inputDepth, const uchar3 *rgb, bool isKeyFrame)
{
    _fusion->processFrame(_frame,inputDepth,rgb,isKeyFrame);
}


sMatrix4 FusionHandler::getPose() const
{
    return _fusion->getPose();
}

FusionHandler::~FusionHandler()
{
    delete _fusion;
}

int FusionHandler::keyFramesNum() const
{
    return _fusion->keyFramesNum();
}

void FusionHandler::dropKeyFrame(int val)
{
    _fusion->dropKeyFrame(val);
}

void FusionHandler::setKeyFramePose(int idx, const sMatrix4 &p)
{
    _fusion->setKeyFramePose(idx,p);
}

sMatrix4 FusionHandler::getLastKFPose() const
{
    _fusion->getLastKFPose();
}

sMatrix4 FusionHandler::getKeyFramePose(int idx) const
{
    _fusion->getKeyFramePose(idx);
}

bool FusionHandler::fuseVolumes()
{
    _fusion->fuseVolumes();
}

bool FusionHandler::fuseLastKeyFrame(sMatrix4 &pose)
{
    _fusion->fuseLastKeyFrame(pose);
}

bool FusionHandler::raycasting(uint frame)
{
    _fusion->raycasting(frame);
}

void FusionHandler::renderImage(uchar3 * out)
{
    _fusion->renderImage(out);
}

void FusionHandler::setPose(const sMatrix4 &pose_)
{
    _fusion->setPose(pose_);
}

void FusionHandler::saveVolume(const char *filename) const
{
    Volume v=_fusion->getVolume();
    int size=params.volume_resolution.x*params.volume_resolution.y*params.volume_resolution.z*sizeof(short2);

    float vsize=params.volume_size.x/params.volume_resolution.x;

    short2 *host_data=new short2[size];
    _fusion->getVolumeData(host_data);    

    saveVoxelsToFile(filename,params.volume_resolution,v.getVoxelSize().x,host_data);

    delete []host_data;
}

void FusionHandler::saveHash(const char *filename) const
{
    Volume v=_fusion->getVolume();
    int hashSize=v.getHashSize();

    tsdfvh::HashEntry *e=new tsdfvh::HashEntry[hashSize];

    v.saveHash(e);
    std::ofstream outFile(filename, std::ios::out);
    for(int i=0;i<hashSize;i++)
    {
        outFile<<"("<<e[i].position.x<<","<<
                      e[i].position.y<<","<<
                      e[i].position.z<<"):"<<e[i].next_ptr<<"\n";
    }
    outFile.close();
}

