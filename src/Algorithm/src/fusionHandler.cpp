#include "fusionHandler.h"
#include<iostream>
#include"utils.h"

#include"defs.h"
#include"constant_parameters.h"
#include <unistd.h>
// #include"kernelscalls.h"


FusionHandler::FusionHandler(const kparams_t &p,sMatrix4 initPose)
    :params(p),
     _frame(-1) 
{
    _fusion = new KFusion(params,initPose);    
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
