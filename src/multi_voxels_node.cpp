#include <ros/ros.h>

//#include<swap>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>

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

// #include<fusionHandler.h>
// #include<kfusion.h>
#include<kparams.h>
#include<kfusion.h>
#include<volume.h>

#include<defs.h>

// #include <opencv2/core/core.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>

#include<key_frame_publisher/boolStamped.h>

#define CAM_INFO_TOPIC "/camera/depth/camera_info"
#define RGB_TOPIC "/camera/rgb/image_rect_color"
#define DEPTH_TOPIC "/camera/depth/image_rect"
#define KEY_FRAME_TOPIC "/isKeyFrame"
#define PUB_ODOM_PATH_TOPIC "/multi_voxels/odom_path"
#define DROP_KF_TOPIC "/drop_key_frame"

#define PUB_VOLUME_TOPIC "/multi_voxels/volume_rendered"
#define PUB_ODOM_TOPIC "/multi_voxels/odom"
#define PUB_POINTS_TOPIC "/multi_voxels/pointCloud"
#define OPT_PATH_TOPIC "/odom/path"

#define DEPTH_FRAME "camera_rgb_optical_frame"

#define VO_FRAME "odom"
#define ODOM_FRAME "odom"
#define BASE_LINK "base_link"

#define PUBLISH_POINT_RATE 10
#define PUBLISH_IMAGE_RATE 1

typedef unsigned char uchar;


kparams_t params;
KFusion *fusion=nullptr;
int frame = -1;

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

//ros publishers
ros::Publisher volume_pub;
ros::Publisher odom_pub;
ros::Publisher odom_path_pub;
ros::Publisher points_pub;
ros::Publisher key_frame_pub;
ros::Publisher harris_pub;
ros::Publisher pcl_pub0, pcl_pub1;
sensor_msgs::PointCloud pcl_msg0, pcl_msg1;

//odom path
nav_msgs::Path odomPath;

//frames
std::string depth_frame,vo_frame,base_link_frame,odom_frame;

//functions
void publishVolumeProjection();
void publishOdom();
void publishOdomPath(geometry_msgs::Pose &p);
void publishPoints();
sMatrix4 homoFromRosPose(const geometry_msgs::Pose &p);



sMatrix4 homoFromRosPose(const geometry_msgs::Pose &p)
{
    sMatrix4 ret;
    //ROS is stupid
    tf::Quaternion q(p.orientation.x,
                     p.orientation.y,
                     p.orientation.z,
                     p.orientation.w);
    
    
    tf::Matrix3x3 rot(q);
    
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
             ret(i,j)=rot[i][j];
        }
    }
    ret(0,3)=p.position.x;
    ret(1,3)=p.position.y;
    ret(2,3)=p.position.z;
    ret(3,3)=1;        
    return ret;
    
}

void imageAndDepthCallback(const sensor_msgs::ImageConstPtr &rgb,
                           const sensor_msgs::ImageConstPtr &depth,
                           const key_frame_publisher::boolStampedConstPtr &keyFrame)
{    
    if(rgb->header.seq != depth->header.seq || rgb->header.seq != keyFrame->header.seq)
    {
        ROS_ERROR("Wrong sequence numbers.");
        return;
    }
    
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
    if(strcmp(depth->encoding.c_str(), "32FC1")!=0)
    {
        ROS_ERROR("Not supported depth format.");
    }
    
    //frame++;
    frame=depth->header.seq;
    
    memcpy(inputRGB,cv_ptr->image.data ,params.inputSize.y*params.inputSize.x*sizeof(uchar)*3 );
    memcpy(inputDepthFl,depth->data.data(),params.inputSize.y*params.inputSize.x*sizeof(float) );
    fusion->processFrame(frame, inputDepthFl, inputRGB, keyFrame->indicator.data);
    
    publishOdom();

    if(publish_volume)
    {
        publishVolumeProjection();
    }
    
    
    if(publish_points && frame % publish_points_rate ==0)
         publishPoints();    
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
    inputDepthFl=new float[params.inputSize.x * params.inputSize.y];
    
    if(publish_volume)    
        volumeRender = new uchar3[params.computationSize.x * params.computationSize.y];    
    else
        volumeRender=nullptr;
    fusion=new KFusion(params,poseMatrix);
}

void dropKeyFrameCb(const std_msgs::Int32 &msg)
{
    int val=msg.data;
    std::cout<<"Dropping key frame:"<<val<<std::endl;
    fusion->dropKeyFrame(val);
}

void optimizedPathCb(const nav_msgs::Path &msg)
{
    ROS_INFO("Got optimized poses");
    if(fusion->keyFramesNum() != msg.poses.size())
    {
        ROS_ERROR("Got %ld poses but %d is expected.",msg.poses.size(),fusion->keyFramesNum());
        //return;
    }
    
    for(int i=0;i<msg.poses.size()-1;i++)
    {
        geometry_msgs::Pose pose=msg.poses[i].pose;
        sMatrix4 p=homoFromRosPose(pose);
        
        p(0,3)+=params.volume_direction.x;
        p(1,3)+=params.volume_direction.y;
        p(2,3)+=params.volume_direction.z;
        
        sMatrix4 prevPose=fusion->getKeyFramePose(i);
        
        fusion->setKeyFramePose(i,p);
        sMatrix4 delta = inverse(p)*prevPose;
//         std::cout<<prevPose<<std::endl;
//         std::cout<<p<<std::endl;
//         std::cout<<delta<<std::endl;
        
//         std::cout<<std::endl;
    }
    
    geometry_msgs::Pose lastPose=msg.poses.back().pose;
    sMatrix4 lastPoseMat=homoFromRosPose(lastPose);
    
    lastPoseMat(0,3)+=params.volume_direction.x;
    lastPoseMat(1,3)+=params.volume_direction.y;
    lastPoseMat(2,3)+=params.volume_direction.z;
    
    sMatrix4 lastKFpose=fusion->getLastKFPose();
    sMatrix4 kfpose=fusion->getPose();
    
    sMatrix4 delta=inverse(lastKFpose)*kfpose;
    sMatrix4 newKfPose=lastPoseMat*delta;
    
#ifdef SAVE_VOXELS_TO_FILE
        Volume vol=fusion->getVolume();
        char buf[64];
        sprintf(buf,"/tmp/voxels/f%d_voxels",frame);
        saveVoxelsToFile(buf,vol);
#endif        

        ROS_INFO("Fusing volumes");
        fusion->fuseVolumes();            
        fusion->fuseLastKeyFrame(lastPoseMat);
        fusion->setPose(newKfPose);
        
        
#ifdef SAVE_VOXELS_TO_FILE            
        vol=fusion->getVolume();            
        sprintf(buf,"/tmp/voxels/f%d_voxels",frame+1);
        saveVoxelsToFile(buf,vol);
#endif 
}

void publishVolumeProjection()
{    
    fusion->renderImage(volumeRender);
    
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
    sMatrix4 pose = fusion->getPose();

    pose(0,3)-=params.volume_direction.x;
    pose(1,3)-=params.volume_direction.y;
    pose(2,3)-=params.volume_direction.z;
    //pose=fromVisionCord(pose);

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
    std::vector<float3> vertices;
    fusion->getVertices(vertices);
    
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

    std::string cam_info_topic,depth_topic,rgb_topic,key_frame_topic,opt_path_topic,drop_kf_topic;    

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
    if(!n_p.getParam("key_frame_topic", key_frame_topic))
    {
        key_frame_topic=std::string(KEY_FRAME_TOPIC);
    }
    if(!n_p.getParam("opt_path_topic", opt_path_topic))
    {
        opt_path_topic=std::string(OPT_PATH_TOPIC);
    }
    if(!n_p.getParam("drop_kf_topic", drop_kf_topic))
    {
        drop_kf_topic=std::string(DROP_KF_TOPIC);
    }
    

    n_p.param("publish_volume",publish_volume,true);    
    n_p.param("publish_points",publish_points,false);
    n_p.param("publish_points_rate",publish_points_rate, PUBLISH_POINT_RATE);        

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

    ros::Subscriber optimized_path_sub = n_p.subscribe(opt_path_topic, 10, optimizedPathCb);
    ros::Subscriber drop_kf_subsub = n_p.subscribe(drop_kf_topic, 10, dropKeyFrameCb);

    
    ROS_INFO("Waiting depth message");

    
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(n_p, rgb_topic, 100);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(n_p, depth_topic, 100);
    message_filters::Subscriber<key_frame_publisher::boolStamped> key_frame_sub(n_p, key_frame_topic, 100);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, key_frame_publisher::boolStamped> MySyncPolicy;


    message_filters::Synchronizer<MySyncPolicy> sync( MySyncPolicy(100), rgb_sub, depth_sub,key_frame_sub);

    sync.registerCallback(boost::bind(&imageAndDepthCallback, _1, _2, _3));
      
    ros::spin();
}
