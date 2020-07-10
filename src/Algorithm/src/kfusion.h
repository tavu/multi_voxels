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

        void restorePose()
        {
            pose=oldPose;
        }

        void languageSpecificConstructor();
        ~KFusion();

        void reset();

        bool preprocessing(const ushort * inputDepth,const uchar3 *rgb);
        bool preprocessing2(const float *inputDepth,const uchar3 *rgb) ;

        bool tracking(uint frame);
        bool raycasting(uint frame);
        bool integration(uint frame);

        void getImageProjection(sMatrix4 pose, uchar3 *out);
        float compareRgb();
        float compareRgbTruth(sMatrix4 pose,uchar3 *out );
        void dumpVolume(const  char * filename);
        void renderVolume(uchar3 * out);
        void renderImage(uchar3 * out);

        void renderTrack(uchar3 * out);
        void renderDepth(uchar3 * out);
        void renderDepthFromVertex(uchar3 * out);

        void getVertices(std::vector<float3> &vertices);

        float getFitness();
        sMatrix4 getPose() const
        {
            return pose;
        }

        sMatrix4 getPoseInv() const
        {
            return poseInv;
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


        float calcNewMapInfo(uchar3 *out);

        Volume& getVolume()
        {
            return volume;
        }

        Volume getNewDataVolume()
        {
            return newDataVol;
        }

        void integrateNewData(sMatrix4 p);
        bool deIntegration(sMatrix4 p,const Host &depth,const Host &rgb);
        bool reIntegration(sMatrix4 pose,const Host &depth,const Host &rgb);
        void getImageRaw(RgbHost &to) const;
        void getDepthRaw(DepthHost &data) const;
        
        
        void getIcpValues(Image<float3, Host> &depthVertex,
                          Image<float3, Host> &raycastVertex,
                          Image<float3, Host> &normals,
                          Image<TrackData, Host> &trackData) const;

        sMatrix6 calculate_ICP_COV();
        
        Image<float3, Host> getAllVertex();
        Image<float3, Host> getAllNormals();

        Image<TrackData, Host> getTrackData();
        float getWrongNormalsSize();
        
        Image<float, Host> vertex2Depth();
        void updateVolume();
        void integrateSlices(VolumeSlices &slices);


    private:
        bool _tracked;
        bool forcePose;
        float step;
        sMatrix4 pose;
        sMatrix4 poseInv;
        sMatrix4 oldPose;
        sMatrix4 deltaPose;
        sMatrix4 trackPose;
        sMatrix4 oldRaycastPose;
        sMatrix4 *viewPose;
        sMatrix4 inverseCam;
        sMatrix4 camMatrix;
        std::vector<int> iterations;
        Volume volume;
        float largestep;
        Volume newDataVol;

        const kparams_t &params;

        sMatrix4 raycastPose;

        Image<TrackData, Device> reduction;
        Image<float3, Device> vertex, normal;
        std::vector<Image<float3, Device> > inputVertex, inputNormal;
        std::vector<Image<float, Device> > scaledDepth;
        
        Image<sMatrix6, Device> covData;

        Image<float, Device> rawDepth;
        Image<ushort, Device> depthImage;
        Image<uchar3, Device> rawRgb;
        Image<float, HostDevice> output;
        Image<float, Device> gaussian;

        Image<uchar3, Device> renderModel;
//         Image<uchar3, HostDevice>  trackModel, depthModel;




//        Match3D match3d;

        //Functions
        bool updatePoseKernel(sMatrix4 & pose, const float * output,float icp_threshold,sMatrix4 &deltaPose);
        bool checkPoseKernel(sMatrix4 & pose,
                             sMatrix4 oldPose,
                             const float * output,
                             uint2 imageSize,
                             float track_threshold);
};
#endif
