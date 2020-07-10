#include <ros/ros.h>

//#include<swap>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>

#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <string.h>
#include <kernels.h>

#include <tf/LinearMath/Matrix3x3.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// #include <actionlib/client/simple_action_client.h>

#include<fusionHandler.h>
#include<kparams.h>
#include<kfusion.h>
#include<volume.h>

#include<defs.h>

// #include <opencv2/core/core.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// #include <std_srvs/SetBool.h>

#define CAM_INFO_TOPIC "/camera/depth/camera_info"
#define RGB_TOPIC "/camera/rgb/image_rect_color"
#define DEPTH_TOPIC "/camera/depth/image_rect"

#define PUB_VOLUME_TOPIC "/multi_voxels/volume_rendered"
#define PUB_ODOM_TOPIC "/multi_voxels/odom"
#define PUB_POINTS_TOPIC "/multi_voxels/pointCloud"

#define DEPTH_FRAME "camera_rgb_optical_frame"

#define VO_FRAME "odom"
#define ODOM_FRAME "odom"
#define BASE_LINK "base_link"

#define PUBLISH_POINT_RATE 10
#define PUBLISH_IMAGE_RATE 1

typedef unsigned char uchar;


kparams_t params;
FusionHandler *kfHandler=nullptr;
int frame = 0;

//Buffers
uint16_t *inputDepth=0;
float *inputDepthFl=0;
uchar3 *inputRGB;
uchar3 *volumeRender;

//other params
bool publish_volume=true;
bool publish_points=true;
bool publish_key_frame=true;

bool publish_key_points = true;

int publish_points_rate;
int key_frame_thr;
int keypt_size;

bool use_bag;

//ros publishers
ros::Publisher volume_pub;
ros::Publisher odom_pub ;
ros::Publisher points_pub;
ros::Publisher key_frame_pub;
ros::Publisher harris_pub;
ros::Publisher pcl_pub0, pcl_pub1;
sensor_msgs::PointCloud pcl_msg0, pcl_msg1;
//frames
std::string depth_frame,vo_frame,base_link_frame,odom_frame;

bool keyFrameProcessing=false;
//functions
void publishVolumeProjection();
void publishOdom();
void publishPoints();

int leftFeetValue=0;
int rightFeetValue=0;
int doubleSupport=0;
int passedFromLastKeyFrame=0;

//keypts vertex
std::vector<float3> keyVert;
std::vector<float3> prevKeyVert;

//poses
sMatrix4 keyFramePose;
sMatrix4 prevKeyFramePose;

//visualization data
uchar3 *featImage;

geometry_msgs::Pose transform2pose(const geometry_msgs::Transform &trans);


#define PUB_ODOM_PATH_TOPIC "/multi_voxels/odom_path"
nav_msgs::Path odomPath, isamPath;
ros::Publisher odom_path_pub ;
void publishOdomPath(geometry_msgs::Pose &p);


void publishFeatures();
void publishKeyPoints();

void stopBag();
void contBag();

// geometry_msgs::Pose rosPoseFromHomo(sMatrix4 &pose)
// {
//     geometry_msgs::Pose ret;
// 
//     pose(0,3)-=params.volume_direction.x;
//     pose(1,3)-=params.volume_direction.y;
//     pose(2,3)-=params.volume_direction.z;
//     pose=fromVisionCord(pose);
// 
//     tf::Matrix3x3 rot_matrix( pose(0,0),pose(0,1),pose(0,2),
//                               pose(1,0),pose(1,1),pose(1,2),
//                               pose(2,0),pose(2,1),pose(2,2) );
//     //rot_matrix=rot_matrix.inverse ();
//     tf::Quaternion q;
//     rot_matrix.getRotation(q);
// 
//     ret.position.x=pose(0,3);
//     ret.position.y=pose(1,3);
//     ret.position.z=pose(2,3);
//     ret.orientation.x=q.getX();
//     ret.orientation.y=q.getY();
//     ret.orientation.z=q.getZ();
//     ret.orientation.w=q.getW();
//     return ret;
// }

geometry_msgs::Pose transform2pose(const geometry_msgs::Transform &trans)
{
    geometry_msgs::Pose pose;
    pose.position.x=trans.translation.x;
    pose.position.y=trans.translation.y;
    pose.position.z=trans.translation.z;

    pose.orientation.x=trans.rotation.x;
    pose.orientation.y=trans.rotation.y;
    pose.orientation.z=trans.rotation.z;
    pose.orientation.w=trans.rotation.w;

    return pose;
}

void imageAndDepthCallback(const sensor_msgs::ImageConstPtr &rgb,const sensor_msgs::ImageConstPtr &depth)
{    
    passedFromLastKeyFrame++;
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge RGB exception: %s", e.what());
        return;
    }
    memcpy(inputRGB,cv_ptr->image.data ,params.inputSize.y*params.inputSize.x*sizeof(uchar)*3 );
    
    if(strcmp(depth->encoding.c_str(), "32FC1")==0) //32FC1
    {
        if(inputDepthFl==0)
            inputDepthFl=new float[params.inputSize.x * params.inputSize.y];
            
        memcpy(inputDepthFl,depth->data.data(),params.inputSize.y*params.inputSize.x*sizeof(float) );
        kfHandler->preprocess(inputDepthFl,inputRGB);
    }
    else if(strcmp(depth->encoding.c_str(), "16UC1")==0) //16UC1
    {
        if(inputDepth==0)
            inputDepth = new uint16_t[params.inputSize.x * params.inputSize.y];   
        
        memcpy(inputDepth,depth->data.data(),params.inputSize.y*params.inputSize.x*2);
        kfHandler->preprocess(inputDepth,inputRGB);
    }
    else
    {
        ROS_ERROR("Not supported depth format.");
        return;
    }
    

    kfHandler->processFrame();
    publishOdom();

    if(publish_volume)
    {
        publishVolumeProjection();
    }
    
    
    if(publish_points && frame % publish_points_rate ==0)
         publishPoints();
    
    frame++;
}

void camInfoCallback(sensor_msgs::CameraInfoConstPtr msg)
{
    params.camera =  make_float4(msg->K[0],msg->K[4],msg->K[2],msg->K[5]);
    params.inputSize.y=msg->height;
    params.inputSize.x=msg->width;
    
    params.computationSize = make_uint2(
                params.inputSize.x / params.compute_size_ratio,
                params.inputSize.y / params.compute_size_ratio);

    
    ROS_INFO("camera is = %f, %f, %f, %f",
             params.camera.x,
             params.camera.y,
             params.camera.z,
             params.camera.w);
    
    sMatrix4 poseMatrix;  
    poseMatrix(0,3)=params.volume_direction.x;
    poseMatrix(1,3)=params.volume_direction.y;
    poseMatrix(2,3)=params.volume_direction.z;
  

    inputRGB = new uchar3[params.inputSize.x * params.inputSize.y];
    if(publish_volume)    
        volumeRender = new uchar3[params.computationSize.x * params.computationSize.y];    
    else
        volumeRender=nullptr;
    kfHandler=new FusionHandler(params,poseMatrix);
}

void publishVolumeProjection()
{
    KFusion *kFusion=kfHandler->fusion();
    kFusion->renderImage(volumeRender);
    
    sensor_msgs::Image image;
    image.header.stamp=ros::Time::now();

    image.width=params.inputSize.x;
    image.height=params.inputSize.y;
    
    int step_size=sizeof(uchar)*3;
    image.is_bigendian=0;
    image.step=step_size*image.width;
    image.header.frame_id=std::string("KFusion_volume");
    image.encoding=std::string("rgb8");

    uchar *ptr=(uchar*)volumeRender;
    image.data=std::vector<uchar>(ptr ,ptr+(params.computationSize.x * params.computationSize.y*step_size) );
    volume_pub.publish(image);
}

void publishOdom()
{
    sMatrix4 pose = kfHandler->getPose();

    pose(0,3)-=params.volume_direction.x;
    pose(1,3)-=params.volume_direction.y;
    pose(2,3)-=params.volume_direction.z;
    pose=fromVisionCord(pose);

    tf::Matrix3x3 rot_matrix( pose(0,0),pose(0,1),pose(0,2),
                              pose(1,0),pose(1,1),pose(1,2),
                              pose(2,0),pose(2,1),pose(2,2) );
    //rot_matrix=rot_matrix.inverse ();
    tf::Quaternion q;
    rot_matrix.getRotation(q);

    nav_msgs::Odometry odom;
    geometry_msgs::Pose odom_pose;
    odom_pose.position.x=pose(0,3);
    odom_pose.position.y=pose(1,3);
    odom_pose.position.z=pose(2,3);
    odom_pose.orientation.x=q.getX();
    odom_pose.orientation.y=q.getY();
    odom_pose.orientation.z=q.getZ();
    odom_pose.orientation.w=q.getW();
    
    //set velocity to zero    
    odom.twist.twist.linear.x = 0;
    odom.twist.twist.linear.y = 0;
    odom.twist.twist.angular.z = 0;
    
    odom.header.stamp = ros::Time::now();    
    odom.header.frame_id = VO_FRAME;
    //odom.child_frame_id = "visual_link";
    odom.child_frame_id = DEPTH_FRAME;


    odom.pose.pose=odom_pose;
    odom_pub.publish(odom);

    publishOdomPath(odom_pose);    
}

void publishOdomPath(geometry_msgs::Pose &p)
{
    geometry_msgs::PoseStamped ps;
    ps.header.stamp = ros::Time::now();
    ps.header.frame_id = VO_FRAME;
    ps.pose=p;
    odomPath.poses.push_back(ps);
    
    nav_msgs::Path newPath=odomPath;
    newPath.header.stamp = ros::Time::now();
    newPath.header.frame_id = VO_FRAME;

    odom_path_pub.publish(newPath);
}

void publishPoints()
{
    KFusion *kFusion=kfHandler->fusion();
    
    std::vector<float3> vertices;
    kFusion->getVertices(vertices);
    
    sensor_msgs::PointCloud pcloud;
    pcloud.header.stamp = ros::Time::now();
    pcloud.header.frame_id = odom_frame;
    pcloud.points.reserve(vertices.size());
    sensor_msgs::ChannelFloat32 ch;    
    
    for(int i=0;i<vertices.size();i++)
    {
        geometry_msgs::Point32 p;
        p.x=vertices[i].x;
        p.y=vertices[i].y;
        p.z=vertices[i].z;

        pcloud.points.push_back(p);
        ch.values.push_back(1);    
    }
    pcloud.channels.push_back(ch);
    points_pub.publish(pcloud);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "multi_voxels_node",ros::init_options::AnonymousName);
    ros::NodeHandle n_p("~");

    std::string cam_info_topic,depth_topic,rgb_topic;    

    if(!n_p.getParam("cam_info_topic", cam_info_topic))
    {
        cam_info_topic=std::string(CAM_INFO_TOPIC);
    }
    if(!n_p.getParam("depth_topic", depth_topic))
    {
        depth_topic=std::string(DEPTH_TOPIC);
    }
    if(!n_p.getParam("rgb_topic", rgb_topic))
    {
        rgb_topic=std::string(RGB_TOPIC);
    }

    n_p.param("publish_volume",publish_volume,true);    
    n_p.param("publish_points",publish_points,false);
    n_p.param("publish_points_rate",publish_points_rate,PUBLISH_POINT_RATE);        

    ROS_INFO("Depth Frame:%s",depth_frame.c_str());      
    
    if(publish_volume)
        volume_pub = n_p.advertise<sensor_msgs::Image>(PUB_VOLUME_TOPIC, 1000);

    if(publish_points)
        points_pub = n_p.advertise<sensor_msgs::PointCloud>(PUB_POINTS_TOPIC, 100);


    odom_pub = n_p.advertise<nav_msgs::Odometry>(PUB_ODOM_TOPIC, 50);
    odom_path_pub = n_p.advertise<nav_msgs::Path>(PUB_ODOM_PATH_TOPIC, 50);

    ROS_INFO("Waiting camera info");
    while(ros::ok())
    {
        sensor_msgs::CameraInfoConstPtr cam_info=ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic);
        if(cam_info)
        {            
            camInfoCallback(cam_info);            
            break;         
        }
    }

    ROS_INFO("Waiting depth message");

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(n_p, rgb_topic, 100);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(n_p, depth_topic, 100);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;


    message_filters::Synchronizer<MySyncPolicy> sync( MySyncPolicy(100), rgb_sub, depth_sub);

    sync.registerCallback(boost::bind(&imageAndDepthCallback, _1, _2));
      
    ros::spin();
}
