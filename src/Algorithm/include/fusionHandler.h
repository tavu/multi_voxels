#ifndef CLOSELOOP_H
#define CLOSELOOP_H

#include"kparams.h"

// #include<vector>
#include"utils.h"

class KFusion;
class FusionHandler
{
    public:
        FusionHandler(const kparams_t &p, sMatrix4 initPose);
        ~FusionHandler();

        bool processFrame(int _frame, const float *inputDepth, const uchar3 *rgb, bool isKeyFrame);
        sMatrix4 getPose() const;
        int keyFramesNum() const;
        void dropKeyFrame(int val);
        void setKeyFramePose(int idx, const sMatrix4 &p);
        sMatrix4 getLastKFPose() const;
        sMatrix4 getKeyFramePose(int idx) const;
        bool fuseVolumes();
        bool fuseLastKeyFrame(sMatrix4 &pose);
        bool raycasting(uint frame);
        void renderImage(uchar3 * out);
        void setPose(const sMatrix4 &pose_);

        void saveVolume(const char *filename) const;

        void saveHash(const char *filename) const;
        
    private:
        KFusion *_fusion;
        const kparams_t &params;
};

#endif // CLOSELOOP_H


