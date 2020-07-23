#include "fusionHandler.h"
#include<iostream>
#include"utils.h"

#include"defs.h"
#include"constant_parameters.h"
#include <unistd.h>
#include"volume.h"

#include"kfusion.h"
#include"tsdfvh/voxel.h"

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

}
