#include "fusionHandler.h"
#include<iostream>
#include"utils.h"

#include"defs.h"
#include"constant_parameters.h"
#include <unistd.h>
#include"volume.h"


FusionHandler::FusionHandler(const kparams_t &p,sMatrix4 initPose)
    :params(p),
     _frame(-1) 
{
    _fusion = new KFusion(params,initPose);
    _fusion->initKeyFrame(0);
}


bool FusionHandler::preprocess(uint16_t *depth,uchar3 *rgb)
{
    _fusion->preprocessing(depth,rgb);
    return true;
}

bool FusionHandler::preprocess(float *depth,uchar3 *rgb)
{
    _fusion->preprocessing2(depth,rgb);
    return true;
}


bool FusionHandler::processFrame()
{
    _frame++;
    std::cout<<"[FRAME="<<_frame<<"]"<<std::endl;

    tracked=_fusion->tracking(_frame);
    bool integrated=_fusion->integration(_frame);

    if(!tracked)
    {
        std::cerr<<"[FRAME="<<_frame<<"] Tracking faild!"<<std::endl;
    }
    if(!integrated)
    {
        std::cerr<<"[FRAME="<<_frame<<"] Integration faild!"<<std::endl;        
    }
    else
    {
        _fusion->integrateKeyFrameData();
    }

    if(_frame>0 && (_frame % KEY_FRAME_THR)==0)
    {
        _fusion->initKeyFrame(_frame);
        
        if(_fusion->keyFramesNum()==MAX_KEY_FRAMES)
        {

#ifdef SAVE_VOXELS_TO_FILE
            Volume vol=_fusion->getVolume();
            char buf[64];
            sprintf(buf,"/tmp/voxels/f%d_voxels",_frame);
            saveVoxelsToFile(buf,vol);
#endif        
            _fusion->fuseVolumes();

            #ifdef SAVE_VOXELS_TO_FILE            
            vol=_fusion->getVolume();            
            sprintf(buf,"/tmp/voxels/f%d_voxels",_frame+1);
            saveVoxelsToFile(buf,vol);
#endif

#ifdef SAVE_VOLUMES_FRAME    
            _fusion->saveVolumes((char*)"/tmp/voxels");
#endif   
        }
    }

    bool raycast=_fusion->raycasting(_frame);
    if(!raycast)
    {
        std::cerr<<"[FRAME="<<_frame<<"] Raycast faild!"<<std::endl;
    }

    return tracked;
}

sMatrix4 FusionHandler::getPose() const
{
    return _fusion->getPose();
}

FusionHandler::~FusionHandler()
{
    delete _fusion;
}
