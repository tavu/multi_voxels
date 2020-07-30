#ifndef CLOSELOOP_H
#define CLOSELOOP_H

#include"kparams.h"

// #include<vector>
#include"utils.h"

/**
* @brief      Class handling KFusion.
*             The implementation of KFusion is in cuda.
*             FusionHandler use solely cpp syntax separating cpp and cuda.
*/
class KFusion;
class FusionHandler
{
    public:
        /**
        * @brief      Construct the fusion handler.
        *
        * @param[in]  p The input parameters.
        */
        FusionHandler(const kparams_t &p, sMatrix4 initPose);
        ~FusionHandler();

        /**
        * @brief      Process the given frame
        *
        * @param[in]  _frame frame id.
        *
        * @param[in]  inputDepth the depth of the frame in meters.
        *
        * @param[in]  The rgb data.
        *
        * @param[in]  isKeyFrame true if this is a key frame.
        *
        * @return     Returns true if KFusion successfully tracked the camera
        */
        bool processFrame(int _frame, const float *inputDepth, const uchar3 *rgb, bool isKeyFrame);

        /**
        * @brief      Get camera pose
        * @return     Returns the current pose of the camera in the world.
        */
        sMatrix4 getPose() const;

        /**
        * @brief      The number of key frames in RAM
        * @return     Returns the number of the key frames in the RAM .
        */
        int keyFramesNum() const;

        /**
        * @brief      Removes the key frame of the given idx and frees its data
        * @param[in]  The id of the key frame
        * @return     Returns the number of the key frames.
        */
        void dropKeyFrame(int val);

        /**
        * @brief      Sets the pose of the given key frame.
        *             This poses are used for fusion
        *
        * @param[in]  idx The id of the key frame.
        * @return     p the pose of the key frame in the world
        */
        void setKeyFramePose(int idx, const sMatrix4 &p);

        /**
        * @brief      Gets the position of the last(current) key frame.
        *
        * @return     the position of the last(current) key frame.
        */
        sMatrix4 getLastKFPose() const;

        /**
        * @brief      Gets the position of the key frame with the given id.
        *
        * @param[in]  The id of the requested key frame.
        * @return     the position of the last(current) key frame.
        */
        sMatrix4 getKeyFramePose(int idx) const;

        /**
        * @brief      Fuse all volumes together
        *             After this funcion you may want to call fuseLastKeyFrame
        *
        * @param[in]  The id of the requested key frame.
        * @return     true on success false otherwise.
        */
        bool fuseVolumes();

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
        * @brief      Raycast.
        *
        * @return     true on success false otherwise.
        */
        bool raycasting(uint frame);


        /**
        * @brief       Create an image of the projection of the volume to camera frame based on previous raycast
        *
        * @param[out]  The rgb image
        */
        void renderImage(uchar3 * out);

        /**
        * @brief      Sets the current pose of the camera.
        *
        * @param[in]  The current pose on world frame.
        */
        void setPose(const sMatrix4 &pose_);

        /**
        * @brief      saves the volume on the disk.
        *
        * @param[in]  filename The filename(full or relative path) of the output file.
        */
        void saveVolume(const char *filename) const;

        /**
        * @brief      saves the hash table on the disk.
        *
        * @param[in]  filename The filename(full or relative path) of the output file.
        */
        void saveHash(const char *filename) const;


        /**
        * @brief      Get the id of the last key frame (current key frame)
        *
        * @return    the id of the last key frame (current key frame).
        */
        int getLastKeyFrameIdx() const;

        
    private:
        KFusion *_fusion;
        const kparams_t &params;
};

#endif // CLOSELOOP_H


