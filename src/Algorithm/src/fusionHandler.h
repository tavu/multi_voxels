#ifndef CLOSELOOP_H
#define CLOSELOOP_H

#include"kfusion.h"
#include"kparams.h"
// #include"Isam.h"

// #include<vector>
#include"utils.h"
// #include"featuredetector.h"
// #include"keyptsmap.h"

// #include"PoseGraph.h"
// #include<list>

// #include"Isam.h"

// #include"featuredetector.h"

class FusionHandler
{
    public:
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        FusionHandler(const kparams_t &p, sMatrix4 initPose);
        ~FusionHandler();

        bool preprocess(uint16_t *depth,uchar3 *rgb);
        bool preprocess(float *depth,uchar3 *rgb);

        bool processFrame();

        KFusion* fusion() const
        {
            return _fusion;
        }

        sMatrix4 getPose() const;


    private:
//         
        KFusion *_fusion;

        const kparams_t &params;
        int _frame;
  
        bool tracked;

};

#endif // CLOSELOOP_H


