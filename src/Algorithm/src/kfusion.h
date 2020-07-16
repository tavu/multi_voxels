#ifndef ICS_FUSION_H
#define ICS_FUSION_H

#include"kparams.h"
#include"utils.h"
#include"volume.h"
#include<vector>
#include<iostream>

class KFusion
{
    public:
      
        //Allow a kfusion object to be created with a pose which include orientation as well as position
        KFusion(const kparams_t &par,sMatrix4 initPose);

        ~KFusion();

        void reset();
        
        bool processFrame(int _frame, const float *inputDepth, const uchar3 *rgb, bool isKeyFrame);

        bool preprocessing(const ushort * inputDepth,const uchar3 *rgb);
        bool preprocessing(const float *inputDepth,const uchar3 *rgb) ;

        bool tracking(uint frame);
        bool raycasting(uint frame);
        bool integration(uint frame);

        void getImageProjection(sMatrix4 pose, uchar3 *out);
        void renderImage(uchar3 * out);

        void renderTrack(uchar3 * out);
        void renderDepth(uchar3 * out);
        void renderDepthFromVertex(uchar3 * out);

        void getVertices(std::vector<float3> &vertices);
        sMatrix4 getPose() const
        {
            return pose;
        }

        void setPose(const sMatrix4 pose_)
        {
            pose=pose_;
            forcePose=true;
            _tracked=true;
        }
        void setViewPose(sMatrix4 *value = NULL)
        {
            if (value == NULL)
                viewPose = &pose;
            else
                viewPose = value;
        }
        
        sMatrix4 *getViewPose()
        {
            return (viewPose);
        }

        Volume& getVolume()
        {
            return volume;
        }

        Volume getKeyFrameVolume()
        {
            return keyFrameVol;
        }

        void setKeyFramePose(int idx, const sMatrix4 &p)
        {
            volumes[idx].pose=p;
        }
        
        sMatrix4 getKeyFramePose(int idx) const
        {
            return volumes[idx].pose;
        }
        
        void integrateKeyFrameData();
        bool deIntegration(sMatrix4 p,const Host &depth,const Host &rgb);
        bool reIntegration(sMatrix4 pose,const Host &depth,const Host &rgb);

        Image<TrackData, Host> getTrackData();

        bool initKeyFrame(uint frame);

        void saveVolumes(char *dir);
        bool fuseVolumes();
        
        uint keyFramesNum() const
        {
            return volumes.size();
        }
        
        void clearKeyFramesData();
        
    private:
        int _frame;
        bool _tracked;
        bool forcePose;
        float step;
        sMatrix4 pose;

        sMatrix4 oldPose;
        sMatrix4 deltaPose;
        sMatrix4 trackPose;

        sMatrix4 lastKeyFramePose;
        uint lastKeyFrameIdx;
        
        sMatrix4 *viewPose;
        sMatrix4 inverseCam;
        sMatrix4 camMatrix;
        std::vector<int> iterations;
        Volume volume;
        float largestep;
        Volume keyFrameVol;

        const kparams_t &params;

        sMatrix4 raycastPose;

        Image<TrackData, Device> reduction;
        Image<float3, Device> vertex, normal;
        std::vector<Image<float3, Device> > inputVertex, inputNormal;
        std::vector<Image<float, Device> > scaledDepth;

        Image<float, Device> rawDepth;
        Image<uchar3, Device> rawRgb;
        
        Image<ushort, Device> depthImage;

        Image<float, HostDevice> output;
        Image<float, Device> gaussian;

        Image<uchar3, Device> renderModel;

        std::vector<VolumeCpu> volumes;

        
        //Functions
        bool updatePoseKernel(sMatrix4 & pose, const float * output,float icp_threshold,sMatrix4 &deltaPose);
        bool checkPoseKernel(sMatrix4 & pose,
                             sMatrix4 oldPose,
                             const float * output,
                             uint2 imageSize,
                             float track_threshold);
};
#endif
