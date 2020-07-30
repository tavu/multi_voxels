#ifndef ICS_FUSION_H
#define ICS_FUSION_H

#include"kparams.h"
#include"sMatrix.h"
#include"volume.h"
#include<vector>
#include<iostream>
#include"image.h"

class KFusion
{
    public:
      
        /**
        * @brief      Construct the fusion handler.
        *
        * @param[in]  p The input parameters.
        * @param[in]  p The initial position of the camera on world
        */
        KFusion(const kparams_t &par,sMatrix4 initPose);

        ~KFusion();

        /**
        * @brief      Process the given frame
        *
        * @param[in]  _frame frame id.
        *
        * @param[in]  inputDepth the depth of the frame in meters.
        *
        * @param[in]  rgb   The rgb data.
        *
        * @param[in]  isKeyFrame true if this is a key frame.
        *
        * @return     Returns true if KFusion successfully tracked the camera
        */
        bool processFrame(int _frame,
                          const float *inputDepth,
                          const uchar3 *rgb,
                          bool isKeyFrame);

        /**
        * @brief      Converts depth to meters,
        *             applies a bilater filter to depth
        *             and stores the input data internally
        *
        * @param[in]  rgb   The rgb data.
        * @param[in]  inputDepth   The depth data.
        * @return     True on success False otherwise
        */
        bool preprocessing(const ushort * inputDepth,const uchar3 *rgb);


        /**
        * @brief      Applies bilater filter to depth
        *             and stores the input data internally
        *
        * @param[in]  rgb   The rgb data.
        * @param[in]  inputDepth   The depth data.
        * @return     True on success False otherwise
        */
        bool preprocessing(const float *inputDepth,const uchar3 *rgb) ;

        /**
        * @brief      Camera tracking
        * @param[in]  frame The id of the given frame
        * @return     True on success False otherwise
        */
        bool tracking(uint frame);

        /**
        * @brief      Raycasting
        * @param[in]  frame The id of the given frame
        * @return     True on success False otherwise
        */
        bool raycasting(uint frame);

        /**
        * @brief      Integrates data into the volume
        * @param[in]  frame The id of the given frame
        * @return     True on success False otherwise
        */
        bool integration(uint frame);

        /**
        * @brief      Gets the projection of the map to camera model
        *             positioning on pose.
        * @param[in]  pose The position of the camera
        * @param[in]  The output data (rgb)
        */
        void getImageProjection(sMatrix4 pose, uchar3 *out);

        /**
        * @brief       Create an image of the projection of the volume to camera frame based on previous raycast
        *
        * @param[out]  The RGB image
        */
        void renderImage(uchar3 * out);

        /**
        * @brief       Visualization of tracking
        *
        * @param[out]  The rgb image
        */
        void renderTrack(uchar3 * out);

        /**
        * @brief       RGB visualization of raw depth data
        *
        * @param[out]  The rgb image
        */
        void renderDepth(uchar3 * out);

        /**
        * @brief       RGB visualization of depth vertices
        *
        * @param[out]  The rgb image
        */
        void renderDepthFromVertex(uchar3 * out);

        /**
        * @brief      Get camera pose
        * @return     Returns the current pose of the camera in the world.
        */
        sMatrix4 getPose() const
        {
            return pose;
        }

        /**
        * @brief      Sets the current pose of the camera.
        *
        * @param[in]  The current pose on world frame.
        */
        void setPose(const sMatrix4 &pose_)
        {
            pose=pose_;
            forcePose=true;
            _tracked=true;
        }

//        /**
//        * @brief      Sets the raycast pose
//        * @param[in]  The raycast pose on world frame.
//        */
//        void setViewPose(sMatrix4 *value = NULL)
//        {
//            if (value == NULL)
//                viewPose = &pose;
//            else
//                viewPose = value;
//        }
        
//        /**
//        * @brief      Get raycast pose
//        * @return     Returns the raycast pose in the world cordinates.
//        */
//        sMatrix4 *getViewPose()
//        {
//            return (viewPose);
//        }

        /**
        * @brief      Gets the global volume.
        * @return     The volume representing the map.
        */
        Volume& getVolume()
        {
            return volume;
        }

        /**
        * @brief      Get the volume that contains only the data from the previous key frame.
        * @return     The volume representing the map only after the previous key frame.
        */
        Volume getKeyFrameVolume()
        {
            return keyFrameVol;
        }

        /**
        * @brief      Sets the pose of the given key frame.
        *             This poses are used for fusion
        *
        * @param[in]  idx The id of the key frame.
        * @return     p the pose of the key frame in the world
        */
        void setKeyFramePose(int idx, const sMatrix4 &p)
        {
            volumes[idx].pose=p;
        }


        /**
        * @brief      Gets the position of the key frame with the given id.
        *
        * @param[in]  The id of the requested key frame.
        * @return     the position of the last(current) key frame.
        */
        sMatrix4 getKeyFramePose(int idx) const
        {
            return volumes[idx].pose;
        }

        /**
        * @brief      Gets the position of the last key frame.
        * @return     the position of the last key frame.
        */
        sMatrix4 getLastKFPose() const
        {
            return  lastKeyFramePose;
        }
        
        /**
        * @brief      Integrates the data of the current frame
        *             into volume representing the map after the last key frame.
        */
        void integrateKeyFrameData();

        /**
        * @brief      Get the data of tracking.
        * @return     The data of tracking
        */
        Image<TrackData, Host> getTrackData();

        /**
        * @brief      Inits a new key frame.
        *             Data of the previous key frame are stored into RAM.
        *             Then the keyFrameVol is marked as empty.
        * @return     True on success False otherwise
        */
        bool initKeyFrame(uint frame);

        /**
        * @brief      Fuse all volumes together
        *             After this funcion you may want to call fuseLastKeyFrame
        *
        * @param[in]  The id of the requested key frame.
        * @return     true on success false otherwise.
        */
        bool fuseVolumes();
        
        /**
        * @brief      The number of key frames in RAM
        * @return     Returns the number of the key frames in the RAM .
        */
        uint keyFramesNum() const
        {
            return volumes.size();
        }
        
        /**
        * @brief      Empty the listo of key frames and free the memory
        */
        void clearKeyFramesData();

        /**
        * @brief      Removes the key frame of the given idx and frees its data
        * @param[in]  The id of the key frame
        */
        void dropKeyFrame(int val);
        
        /**
        * @brief      Fuse the last key frame's data to volume.
        *             Last key frame data contains an unfinished volume on GPU
        *             This function fuses this data to the main volume
        *
        * @param[in]  pose The pose of the last key frame on world position
        * @return     true on success false otherwise.
        */
        bool fuseLastKeyFrame(sMatrix4 &pose);

        /**
        * @brief        Gets the data of the volume from GPU to RAM
        *               and stores them into a continuous array (cpu_data).
        *               cpu_data should be allocated and sized
        *               volume.getDimensions().x *
        *               volume.getDimensions().y *
        *               volume.getDimensions().y.
        *               This function is only usuful for visualization of the map.
        * @param[out]   cpu_data The output array of the volume (Should have been allocated).
        * @param[in]    volume The volume that its data are copied.
        *
        */
        void getVolumeData(short2 *cpu_data, Volume &volume);

        /**
        * @brief          As getVolumeData(short2 *cpu_data, Volume &volume)
        *                 when volume is the global volume
        * * @param[out]   cpu_data The output array of the volume (Should have been allocated).
        */
        void getVolumeData(short2 *cpu_data);
        
        /**
        * @brief      Get the id of the last key frame (current key frame)
        * @return    the id of the last key frame (current key frame).
        */
        int getLastKeyFrameIdx() const
        {
            return lastKeyFrameIdx;
        }

        /**
        * @brief     Calculates the covariance of ICP.
        * @return    the covariance matrix.
        */
        sMatrix6 calculate_ICP_COV();

    private:
        /** The sequence number of the current frame **/
        int _frame;
        /** True if the previous tracking succeeded **/
        bool _tracked;
        /** True if the pose has been set with set pose instead of tracking **/
        bool forcePose;
        /** The small step of raycast **/
        float step;
        /** The large step of raycast **/
        float largestep;
        /** The current camera pose the world **/
        sMatrix4 pose;
        /** The previous camera pose the world **/
        sMatrix4 oldPose;

        /** The pose of the last key frame on world coordinates **/
        sMatrix4 lastKeyFramePose;
        /** The id of the last key frame. **/
        uint lastKeyFrameIdx;
        

        /** The camera matrix **/
        sMatrix4 camMatrix;
        /** The inverse of the camera matrix **/
        sMatrix4 inverseCam;

        std::vector<int> iterations;

        /** The global map **/
        Volume volume;
        /** The map with data only after the previous key frame **/
        Volume keyFrameVol;

        /** The id of the last key frame **/
        int lastKeyFrame;
        /** The id of the last frame **/
        int lastFrame;

        /** The parameters of kfusion **/
        const kparams_t &params;

        /** The pose of raycast **/
        sMatrix4 raycastPose;

        /** Data for reduce of tracking **/
        Image<TrackData, Device> reduction;

        /** model vertices and normals **/
        Image<float3, Device> vertex, normal;

        /** input vertices and normals **/
        std::vector<Image<float3, Device> > inputVertex, inputNormal;

        /** depth from bilateral kernel **/
        std::vector<Image<float, Device> > scaledDepth;

        /** raw input depth in meters **/
        Image<float, Device> rawDepth;
        /** raw input RGB **/
        Image<uchar3, Device> rawRgb;
        
        /** raw input depth in mm **/
        Image<ushort, Device> depthImage;

        /** output data of tracking **/
        Image<float, HostDevice> output;

        /** Gaussian of bilateral filter **/
        Image<float, Device> gaussian;

        /** Data of  getImageProjection stored on GPU**/
        Image<uchar3, Device> renderModel;

        /** Volumes stored on RAM **/
        std::vector<VolumeCpu> volumes;

        
        //Functions
         /** Check ICP error and update the pose on every ICP turn **/
        bool updatePoseKernel(sMatrix4 & pose,
                              const float * output,
                              float icp_threshold,
                              sMatrix4 &deltaPose);

        /** Check if tracking succeeded and updates the current pose **/
        bool checkPoseKernel(sMatrix4 & pose,
                             sMatrix4 oldPose,
                             const float * output,
                             uint2 imageSize,
                             float track_threshold);

};
#endif
