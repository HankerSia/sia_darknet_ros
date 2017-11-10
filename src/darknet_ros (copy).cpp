/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>
#include <algorithm>



//#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/search/kdtree.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <sys/time.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types_conversion.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <kinect2_bridge/kinect2_definitions.h>

#include <geometry_msgs/PointStamped.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int8.h>
#include "id_data_msgs/ID_Data.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/core.hpp>

#include "darknet_ros.hpp"
#include <darknet_ros/bbox_array.h>
#include <darknet_ros/bbox.h>

extern "C" {
#include "darknet_src/network.h"
#include "darknet_src/region_layer.h"
#include "darknet_src/cost_layer.h"
//#include "darknet_src/utils.h"
#include "darknet_src/parser.h"
#include "darknet_src/box.h"
#include "darknet_src/demo.h"
#include "darknet_src/option_list.h"
#include "darknet_src/image.h"
}

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointXYZHSV PointHSVType;


cv::Mat cam_image_copy;
ROS_box* output_boxes;

//ros::Subscriber cmd_sub;
//ros::Publisher result_pub;
volatile int DETECT_FLAG = 0;

#define CAR_ADJUST_NUM 1

// define demo_yolo inputs
//char datacfg[] = "/home/ljw/catkin_ws/src/darknet_ros/src/cfg/combine9k.data";
//char cfgfile[] = "/home/ljw/catkin_ws/src/darknet_ros/src/cfg/yolo9000.cfg";
//char weightfile[] = "/home/ljw/workspace/image/yolo9000.weights";

//char datacfg[] = "/home/robot/catkin_ws/src/darknet_ros/src/cfg/coco.data";
//char cfgfile[] = "/home/robot/catkin_ws/src/darknet_ros/src/cfg/tiny-yolo.cfg";
//char weightfile[] = "/home/robot/catkin_ws/src/darknet_ros/src/tiny-yolo.weights";

char datacfg[] = "/home/robot/catkin_ws/src/darknet_ros/src/cfg/single.data";
char cfgfile[] = "/home/robot/catkin_ws/src/darknet_ros/src/cfg/tiny-yolo-single.cfg";
char weightfile[] = "/home/robot/catkin_ws/src/darknet_ros/src/tiny-yolo-single.weights";


const std::string class_labels[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                                     "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
                                     "potted plant", "sheep", "sofa", "train", "tv monitor" };
const int num_classes = sizeof(class_labels)/sizeof(class_labels[0]);

//cv::Mat cam_image_copy;

// define parameters
const std::string CAMERA_COLOR_TOPIC = "/kinect2/qhd/image_color_rect";	
const std::string CAMERA_DEPTH_TOPIC = "/kinect2/qhd/image_depth_rect";
const std::string DETECTION_WINDOW = "detection";
int FRAME_W;
int FRAME_H;
int FRAME_AREA;
int FRAME_COUNT = 0;


network net;
char **names;
image **alphabet;

float thresh;
float hier_thresh;

cv_bridge::CvImagePtr rgb_image;
cv_bridge::CvImagePtr depth_image;

extern "C" int darknet_main(int argc, char **argv);
extern "C" int detector_init(char *datacfg, char *cfgfile, char *weightfile, network* net, char ***names, image ***alphabet);
extern "C" ROS_box* test_detector_ros(network net, image im, char **names, image **alphabet, float thresh, float hier_thresh);
extern "C" float find_float_arg(int argc, char **argv, char *arg, float def);
extern "C" image ipl_to_image(IplImage* src);
//extern "C" void free_image(image m);


image Mat_to_image(cv::Mat& m)
{
    IplImage* src = new IplImage(m);
    image out = ipl_to_image(src);
    return out;
}

cv::Mat image_to_Mat(image p)
{

    image copy = copy_image(p);
    if(p.c == 3) rgbgr_image(copy);
    int x,y,k;

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }

    cv::Mat mt(disp);
    return mt;
}

typedef struct
{
    int u;
    int v;
    unsigned short d;
}Img_ijv;

typedef struct
{
    float x;
    float y;
    float z;
}Obj_loc;

bool Img_ijv_cmp(const Img_ijv& prev, const Img_ijv& next)
{
    return (prev.d < next.d);
}

class Receiver
{
public:
    enum Mode
    {
        IMAGE = 0,
        CLOUD,
        BOTH
    };

private:
    std::mutex lock;
    int car_adjust;

    const std::string topicColor, topicDepth;
    const bool useExact, useCompressed;

    bool updateImage, updateCloud;
    bool save;
    bool running;
    size_t frame;
    const size_t queueSize;

    cv::Mat color, depth;
    cv::Mat cameraMatrixColor, cameraMatrixDepth;
    cv::Mat lookupX, lookupY;

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximateSyncPolicy;

    ros::NodeHandle nh;
    ros::AsyncSpinner spinner;
    image_transport::ImageTransport it;
    ros::Publisher point_pub;
    ros::Subscriber cmd_sub;
    image_transport::SubscriberFilter *subImageColor, *subImageDepth;
    message_filters::Subscriber<sensor_msgs::CameraInfo> *subCameraInfoColor, *subCameraInfoDepth;

    message_filters::Synchronizer<ExactSyncPolicy> *syncExact;
    message_filters::Synchronizer<ApproximateSyncPolicy> *syncApproximate;

    std::thread imageViewerThread;
    Mode mode;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
    pcl::PCDWriter writer;

    std::ostringstream oss;
    std::vector<int> params;

    ros::Publisher _found_object_pub;
    ros::Publisher _bboxes_pub;
    ros::Publisher result_pub;
    std::vector< std::vector<ROS_box> > _class_bboxes;
    std::vector<int> _class_obj_count;
    std::vector<cv::Scalar> _bbox_colors;
    darknet_ros::bbox_array _bbox_results_msg;
    ROS_box* boxes;


    cv::Mat cam2arm;

    int object_col;
    int object_row;

    std::string input_color_file;
    std::string input_depth_file;


public:
    Receiver(const std::string &topicColor, const std::string &topicDepth, const bool useExact, const bool useCompressed)
        : topicColor(topicColor), topicDepth(topicDepth), useExact(useExact), useCompressed(useCompressed),
          updateImage(false), updateCloud(false), save(false), running(false), frame(0), queueSize(5), //5
          nh("~"), spinner(0), it(nh), mode(CLOUD), _class_bboxes(num_classes), _class_obj_count(num_classes, 0), _bbox_colors(num_classes)
    {
        cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);
        cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(100);
        params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        params.push_back(1);
        params.push_back(cv::IMWRITE_PNG_STRATEGY);
        params.push_back(cv::IMWRITE_PNG_STRATEGY_RLE);
        params.push_back(0);

        int incr = floor(255/num_classes);
        for (int i = 0; i < num_classes; i++) {
            _bbox_colors[i] = cv::Scalar(255 - incr*i, 0 + incr*i, 255 - incr*i);
        }

        _found_object_pub = nh.advertise<std_msgs::Int8>("found_object", 1);
        _bboxes_pub = nh.advertise<darknet_ros::bbox_array>("YOLO_bboxes", 1);
        result_pub = nh.advertise<id_data_msgs::ID_Data>("/notice", 1000);

        point_pub = nh.advertise<geometry_msgs::PointStamped>("/notice/targetPoint", 10);

        cam2arm = cv::Mat::zeros(4, 4, CV_64FC1);
        cv::FileStorage fs("/home/robot/catkin_ws/src/darknet_ros/src/cam2arm.yml", cv::FileStorage::READ);
        fs["R_t"] >> cam2arm;
        fs.release();


        cv::namedWindow(DETECTION_WINDOW, cv::WINDOW_NORMAL);

        //spinner.start();
    }

    ~Receiver()
    {
        cv::destroyWindow(DETECTION_WINDOW);
    }

    void run(const Mode mode)
    {
        start(mode);
        stop();
    }

    void ugvReceiveCallback(const id_data_msgs::ID_Data::ConstPtr &msg)
    {
        if(msg->id == 3 &&  msg->data[0]== 1) {
        	ROS_INFO("MSG=======%d\n", msg->id);
            //int flag = msg->data[0];
            object_row = msg->data[1];
            object_col = msg->data[2];
            

            //printf("Need object row=%d, col=%d\n", object_row, object_col);
            DETECT_FLAG = 1;
            car_adjust = CAR_ADJUST_NUM;

            id_data_msgs::ID_Data feedback;
            feedback.id = 3;
            feedback.data[0] = 14;
            result_pub.publish(feedback);
        } 
//        else if (msg->id == 2 &&  msg->data[0]== 8) {
//        	DETECT_FLAG = 1;
//        	car_adjust--;
////        	if(car_adjust == 0)
////        		DETECT_FLAG = 0;
//        }

    }




private:


    void start(const Mode mode)
    {
        this->mode = mode;
        running = true;

        std::string topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
        std::string topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

        image_transport::TransportHints hints(useCompressed ? "compressed" : "raw");
        subImageColor = new image_transport::SubscriberFilter(it, topicColor, queueSize, hints);
        subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, queueSize, hints);
        subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, queueSize);
        subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, queueSize);

        if(useExact)
        {
            syncExact = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
            syncExact->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
        }
        else
        {
            syncApproximate = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
            syncApproximate->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
        }

        cmd_sub = this->nh.subscribe("/notice", 1000, &Receiver::ugvReceiveCallback, this);

        spinner.start();

        std::chrono::milliseconds duration(1);
        while(!updateImage || !updateCloud)
        {
            if(!ros::ok())
            {
                return;
            }
            std::this_thread::sleep_for(duration);
        }

        cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
        cloud->height = color.rows;
        cloud->width = color.cols;
        cloud->is_dense = false;
        cloud->points.resize(cloud->height * cloud->width);

        createLookup(this->color.cols, this->color.rows);

        switch(mode)
        {
        case CLOUD:
            cloudViewer();
            break;
        case IMAGE:
            imageViewer();
            break;
        case BOTH:
            imageViewerThread = std::thread(&Receiver::imageViewer, this);
            cloudViewer();
            break;
        }
    }

    void stop()
    {
        spinner.stop();

        if(useExact)
        {
            delete syncExact;
        }
        else
        {
            delete syncApproximate;
        }

        delete subImageColor;
        delete subImageDepth;
        delete subCameraInfoColor;
        delete subCameraInfoDepth;

        running = false;
        if(mode == BOTH)
        {
            imageViewerThread.join();
        }
    }

    //	void drawBBoxes(cv::Mat &input_frame, std::vector<ROS_box> &class_boxes, int &class_obj_count,
    //		   cv::Scalar &bbox_color, const std::string &class_label)
    //	{
    //		darknet_ros::bbox bbox_result;

    //		for (int i = 0; i < class_obj_count; i++) {
    //			int xmin = (class_boxes[i].x - class_boxes[i].w/2)*FRAME_W;
    //			int ymin = (class_boxes[i].y - class_boxes[i].h/2)*FRAME_H;
    //			int xmax = (class_boxes[i].x + class_boxes[i].w/2)*FRAME_W;
    //			int ymax = (class_boxes[i].y + class_boxes[i].h/2)*FRAME_H;

    //			bbox_result.Class = class_label;
    //			bbox_result.xmin = xmin;
    //			bbox_result.ymin = ymin;
    //			bbox_result.xmax = xmax;
    //			bbox_result.ymax = ymax;
    //			_bbox_results_msg.bboxes.push_back(bbox_result);

    //			// draw bounding box of first object found
    //			cv::Point topLeftCorner = cv::Point(xmin, ymin);
    //			cv::Point botRightCorner = cv::Point(xmax, ymax);
    //			cv::rectangle(input_frame, topLeftCorner, botRightCorner, bbox_color, 2);
    //			cv::putText(input_frame, class_label, cv::Point(xmin, ymax+15), cv::FONT_HERSHEY_PLAIN,
    //			1.0, bbox_color, 2.0);
    //		}
    //	}

    void print_boxes(ROS_box* boxes, int num)
    {
        int i;
        for(i=0; i<num; i++) {
            printf("box#%d\n", i);
            printf("Class:%d, %s\n", boxes[i].Class, names[boxes[i].Class]);
            printf("x:%d\n", boxes[i].x);
            printf("y:%d\n", boxes[i].y);
            printf("w:%d\n", boxes[i].w);
            printf("h:%d\n", boxes[i].h);
            printf("\n");
        }
    }

    cv::Mat& ScanImageAndReduceIterator(cv::Mat& I)
    {
        // accept only char type matrices
        //CV_Assert(I.depth() != sizeof(uchar));
        const int channels = I.channels();
        printf("channels=%d\n",channels);
        switch(channels)
        {
        case 1:
        {
            cv::MatIterator_<unsigned short> it, end;
            for( it = I.begin<unsigned short>(), end = I.end<unsigned short>(); it != end; ++it)
                //*it = table[*it];
                printf("%d ",*it);
            break;
        }
        case 3:
        {
            //		        MatIterator_<Vec3b> it, end;
            //		        for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
            //		        {
            //		            (*it)[0] = table[(*it)[0]];
            //		            (*it)[1] = table[(*it)[1]];
            //		            (*it)[2] = table[(*it)[2]];
            //		        }
            break;
        }
        }
        return I;
    }

//#define SavePcd

    float *GetBoxDistance(pcl::PointCloud<PointType>::Ptr &scene, float x0, float y0, float z0)
    {
        float *results = new float[5];

#ifdef SavePcd
        std::stringstream ss;
        struct timeval tv;
        gettimeofday(&tv, NULL);
        ss << tv.tv_sec << ".pcd";
        std::string filename = ss.str();
        pcl::io::savePCDFile(filename, *scene);
#endif

        pcl::PointCloud<PointHSVType>::Ptr sceneHsv(new pcl::PointCloud<PointHSVType>());
        pcl::PointCloudXYZRGBAtoXYZHSV(*scene, *sceneHsv);

        pcl::IndicesPtr indicesBlue(new std::vector<int>);
        float h, s, v;
        for (int i = 0; i < sceneHsv->points.size(); i++)
        {
            h = sceneHsv->points[i].h;
            s = sceneHsv->points[i].s;
            v = sceneHsv->points[i].v;

            if ((h > 180 && h < 228) && s > 0.17 && s < 1 && v > 0.18 && v < 1)
                indicesBlue->push_back(i);
        }

        pcl::PointCloud<PointType>::Ptr blue(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*scene, *indicesBlue, *blue);

        pcl::PointCloud<PointType>::Ptr inlierBlue(new pcl::PointCloud<PointType>());
        //pcl::RadiusOutlierRemoval<PointType> filter;
        //filter.setInputCloud(blue);
        //filter.setRadiusSearch(0.02);
        //filter.setMinNeighborsInRadius(13);
        //filter.filter(*inlierBlue);

        inlierBlue = blue;

        float x,y,z;
        float deltaX, deltaY, deltaZ;
        float distance;
        float minL = 0.35, minR = 0.35, minT = 0.21, minB = 0.21;
        float maxX = 0;

        pcl::search::KdTree<PointType> tree;
        tree.setInputCloud(inlierBlue);
        std::vector<int> indices(13);
        std::vector<float> squaredDistances(13);

        for (int i = 0; i< inlierBlue->points.size(); i++)
        {
            // Isolated points
            if (tree.nearestKSearch(i, 13, indices, squaredDistances) != 13) continue;
            if (sqrt(squaredDistances[12]) > 0.02) continue;

            x = inlierBlue->points[i].x;
            y = inlierBlue->points[i].y;
            z = inlierBlue->points[i].z;

            // Attention: Please forward 0.1m
            if(x > x0 + 0.1) continue;

            deltaX = x - x0;
            deltaY = y - y0;
            deltaZ = z - z0;

            distance = deltaY * deltaY + deltaZ * deltaZ;
            distance = sqrt(distance);

            if (deltaZ < 0.05 && deltaZ > -0.05 && deltaY > 0 && distance < minL) minL = distance;
            if (deltaZ < 0.05 && deltaZ > -0.05 && deltaY < 0 && distance < minR) minR = distance;
            if (deltaY < 0.05 && deltaY > -0.05 && deltaZ > 0 && distance < minT) minT = distance;
            if (deltaY < 0.05 && deltaY > -0.05 && deltaZ < 0 && distance < minB) minB = distance;

            if(deltaY < 0.05 && deltaY > -0.05 && deltaX < 0 && distance < 0.21 && maxX < -deltaX)
                maxX = -deltaX;
        }

        if(minL + minR < 0.3 || minL + minR > 0.4)
        {
            minL = 0; minR = 0;
        }

        if(minT + minB < 0.16 || minT + minB > 0.26)
        {
            minT = 0; minB = 0;
        }

        if(maxX > 0.5) maxX = 0;

        // TODO: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        // pcl::visualization::PCLVisualizer viewer("Results");
        // viewer.addPointCloud(scene, "Scene");
        //
        // pcl::visualization::PointCloudColorHandlerCustom<PointType> inlierBlueHandler(inlierBlue, 255, 0, 0);
        // viewer.addPointCloud(inlierBlue, inlierBlueHandler, "Blue");
        //
        // while (!viewer.wasStopped()) viewer.spinOnce();
        // TODO: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        results[0] = minL;
        results[1] = minR;
        results[2] = minT;
        results[3] = minB;
        results[4] = maxX;

        return results;
    }

    // just for depth image
    int findNonZeroMinLoc(cv::Mat& image, int *x, int *y)
    {

        int i, j, mx, my, min_value = INT_MAX;
        int nr= image.rows; // number of rows
        int nc= image.cols * image.channels();

        for (j=0; j<nr; j++) {
            unsigned short  *data= image.ptr<unsigned short>(j);
            for (i=0; i<nc; i++) {
                if(min_value > *data && *data > 0) {
                    min_value = *data;
                    mx = i;
                    my = j;
                }
                data++;
            }
        }

        if(min_value != INT_MAX) {
            *x = mx;
            *y = my;
            return 0;
        } else {
            return -1;
        }
    }



    int findNonZeroMiddleLoc(cv::Mat& image, int *u, int *v)
    {
        //cv::Mat o = cv::Mat::zeros(image.rows, image.cols, CV_16UC1);

        int rows = image.rows;
        int cols = image.cols;
        //printf("rows=%d, cols=%d\n", rows, cols);
        Img_ijv d = {0};
        std::vector<Img_ijv> vec;

        for(int i = 0; i < rows; i++) {
            for(int j=0; j < cols; j++) {
                d.d = image.at<unsigned short>(i, j);
                d.u=j;
                d.v=i;
                vec.push_back(d);
            }
        }

        int middle = vec.size()/2;
        //printf("size=%d\n", vec.size());
        //printf("middle=%d\n", middle);
        std::nth_element(vec.begin(), vec.begin()+middle, vec.end(), Img_ijv_cmp);

        *u = vec[middle-1].u;
        *v = vec[middle-1].v;
        //printf("u=%d, v=%d, d=%u\n",*u, *v, vec[middle-1].d);

        return 0;

    }



    float get_depth(cv::Mat& depth_image, cv::Rect& rect)
    {
        float depth;
        cv::Mat im = depth_image(rect);
        //cv::imwrite("depth.png", im);
        //ScanImageAndReduceIterator(im);
        depth =  cv::mean(im)[0];
        //cv::minMaxIdx(image,&min,&max);
        return depth;
    }


    float get_depth_and_center(cv::Mat& depth_image, cv::Rect& rect, int *u, int *v)
    {
        float depth;
        cv::Mat depth_im = depth_image(rect);
        //cv::imwrite("depth.png", depth_im);
        //ScanImageAndReduceIterator(im);
        //depth =  cv::mean(im)[0];
        //int ret = findNonZeroMinLoc(depth_im, u, v);
        int ret = findNonZeroMiddleLoc(depth_im, u, v);
        if(ret < 0) {printf("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr depth get: u=%d, v=%d\n", *u, *v); return -1;}
        *u += rect.x;
        *v += rect.y;
        depth = depth_image.at<unsigned short>(*v, *u) / 1000.0f;
        return depth;
    }

    void UVtoXY(float *x, float *y, int u, int v, float depthValue)
    {
        //		const float ty = lookupY.at<float>(0, v);
        //		const float tx = lookupX.at<float>(0, u);

        //		float fx = 529.973279, fy = 526.966340;
        //		float cx = 477.441620, cy = 261.869293;

        float fx = 505.565471, fy = 506.534290;
        float cx = 461.185699, cy = 278.003533;

        //		printf("tx = %f, ty = %f\n", tx, ty);

        //		*x = tx  * depthValue;
        //		*y = ty  * depthValue;
        *x = (u - cx) * depthValue / fx;
        *y = (v - cy) * depthValue / fy;
    }


    float dxtoDX(int dx, float depthValue)
    {
        float fx = 505.565471;
        return (dx * depthValue / fx);
    }

    float dytoDY(int dy, float depthValue)
    {
        float fy = 506.534290;
        return (dy * depthValue / fy);
    }





    void objectDetect(cv::Mat& color_image, cv::Mat& depth_image)
    {
        image im = Mat_to_image(color_image);
        cvtColor(color_image, color_image, CV_RGB2BGR);

        createCloud(depth_image, color_image,  cloud);
        //    	pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
        //		const std::string cloudName = "rendered";
        //		visualizer->addPointCloud(cloud, cloudName);
        //		visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        //		visualizer->initCameraParameters();
        //		visualizer->setBackgroundColor(0, 0, 0);
        //		visualizer->setPosition(mode == BOTH ? color.cols : 0, 0);
        //		visualizer->setSize(color.cols, color.rows);
        //		visualizer->setShowFPS(true);
        //		visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
        //		visualizer->registerKeyboardCallback(&Receiver::keyboardEvent, *this);
        //		visualizer->updatePointCloud(cloud, cloudName);
        //		for(; running && ros::ok();)
        //		{
        //		  visualizer->spinOnce(10);
        //		}
        //		visualizer->close();

        //float *results = GetBoxDistance(cloud, 0 + 0.05, 0, 0);



        printf("Need obj: row = %d, col = %d\n", object_row, object_col);
        printf("predict start\n");
        boxes = test_detector_ros(net, im, names, alphabet, thresh, hier_thresh);
        //printf("predict over\n");

        // get the number of bounding boxes found
        int num = boxes[0].num;
        printf("############### boxes num: %d######################\n", num);

        //print_boxes(boxes, num);

        int i;
        int u, v;
        float X, Y, Z;

        std::vector<Obj_loc> obj_loc;
        std::vector<Obj_loc> obj_row;

        float robot_height_groud = 0.34;	//unit: m
        float shelf_height_groud = 0.26;
        float shelf_grid_width=0.3467;
        float shelf_grid_height=0.21;
        float shelf_grid_gap = 0.04;

        float bound_left, bound_right, bound_low, bound_high;
        int obj_index=-1;

        //float *results = GetBoxDistance(cloud, obj_loc[obj_index].x, obj_loc[obj_index].y, obj_loc[obj_index].z);

        bound_left = shelf_grid_width/2;
        bound_right = -shelf_grid_width/2;
        bound_low = shelf_height_groud - robot_height_groud + (object_row - 1)*(shelf_grid_height + shelf_grid_gap);
        bound_high = bound_low + shelf_grid_height;

        int valid_count = 0;

        printf("bound: %f, %f, %f, %f\n", bound_right, bound_left, bound_low, bound_high);
        for(i=0; i<num; i++) {
            u = boxes[i].x;
            v = boxes[i].y;

            //			// send message that an object has been detected
            //			 std_msgs::Int8 msg;
            //			 msg.data = 1;
            //			 _found_object_pub.publish(msg);

            int x = boxes[i].x - boxes[i].w/2;
            int y = boxes[i].y - boxes[i].h/2;
            int w = boxes[i].w;
            int h = boxes[i].h;
            cv::Rect rect(x, y, w, h);

            int tu, tv;
            Z = get_depth_and_center(depth_image, rect, &tu, &tv);
            if(Z<=0) {printf("z <=0\n"); continue;}
            UVtoXY(&X, &Y, u, v, Z);

            //convert kinect point to arm point
            cv::Mat kinect_point = cv::Mat::zeros(4, 1, CV_64FC1);
            kinect_point.at<double>(0,0) = X;
            kinect_point.at<double>(1,0) = Y;
            kinect_point.at<double>(2,0) = Z;
            kinect_point.at<double>(3,0) = 1.0f;

            cv::Mat arm_point = cam2arm * kinect_point;

            Obj_loc o;
            o.x = arm_point.at<double>(0,0);
            o.y = arm_point.at<double>(1,0);
            o.z = arm_point.at<double>(2,0);
            obj_loc.push_back(o);


            //if(o.z >= bound_low && o.z<=bound_high && o.y>=bound_right && o.y<=bound_left) obj_index=i;
            if(o.z >= bound_low && o.z<=bound_high && o.y>=bound_right && o.y<=bound_left) obj_index=valid_count;
            

            //if(o.z >= bound_low && o.z<=bound_high) obj_row.push_back(o);

            printf("------ obj#%d ------\n", valid_count);
            //printf("class:%d, %s\n", boxes[i].Class, names[boxes[i].Class]);
            printf("center pixel: u=%d, v=%d\n", u, v);
            printf("cam coord: X=%.2fm, Y=%.2fm, Z=%.2fm\n", X, Y, Z);
            printf("arm coord: X=%.2fm, Y=%.2fm, Z=%.2fm\n", o.x, o.y, o.z);
            printf("\n");
            //			geometry_msgs::PointStamped target_point;
            //		    target_point.header.frame_id = "targetPoint";
            //		    target_point.header.stamp = ros::Time::now();
            //			target_point.point.x = arm_point.at<double>(0,0);
            //			target_point.point.y = arm_point.at<double>(1,0);
            //			target_point.point.z = arm_point.at<double>(2,0);
            //			point_pub.publish(target_point);
            valid_count++;
        }


        //cv::waitKey(0);

        if(obj_index>=0) {
            input_color_file = "color_" + std::to_string(object_row) + std::to_string(object_col) + ".png";
            input_depth_file = "depth_" + std::to_string(object_row) + std::to_string(object_col) + ".png";

            //createCloud(color_image, depth_image, cloud);


            float *results = GetBoxDistance(cloud, obj_loc[obj_index].x, obj_loc[obj_index].y, obj_loc[obj_index].z);


            printf("env info : L=%.2f, R=%.2f, T=%.2f, B=%.2f, X=%.2f\n", results[0], results[1], results[2], results[3], results[4]);

            cv::imwrite("depth_image.png", depth_image);
            cv::imwrite(input_color_file.c_str(), color_image);
            cv::imwrite(input_depth_file.c_str(), depth_image);

            printf("========= Total %d obj find ==========\n", num);
            printf("Need obj: row = %d, col = %d\n", object_row, object_col);
            printf("Result:\n");
            printf("\tobj_index=%d\n", obj_index);
            printf("\tcoord: X=%.2fm, Y=%.2fm, Z=%.2fm\n", obj_loc[obj_index].x, obj_loc[obj_index].y, obj_loc[obj_index].z);
            printf("\tfixed Y=%f\n", ((obj_loc[obj_index].y + results[0]) + (obj_loc[obj_index].y - results[1]))/2.0);

            // tell arm where is object
            id_data_msgs::ID_Data arm_feedback;
            arm_feedback.id = 3;
            arm_feedback.data[0] = 27;
            arm_feedback.data[1] = int(100 * results[0]);
            arm_feedback.data[2] = int(100 * results[1]);
            arm_feedback.data[3] = int(100 * results[2]);
            arm_feedback.data[4] = int(100 * results[3]);
            arm_feedback.data[5] = int(100 * results[4]);
            result_pub.publish(arm_feedback);

            geometry_msgs::PointStamped target_point;
            target_point.header.frame_id = "targetPoint";
            target_point.header.stamp = ros::Time::now();
//            target_point.point.x = obj_loc[obj_index].x;
//            //target_point.point.y = obj_loc[obj_index].y;
//            target_point.point.y = ((obj_loc[obj_index].y + results[0]) + (obj_loc[obj_index].y - results[1]))/2.0;
//            target_point.point.z = obj_loc[obj_index].z;
            target_point.point.x = obj_loc[obj_index].x;
            //target_point.point.y = obj_loc[obj_index].y;
            target_point.point.y = ((obj_loc[obj_index].y + results[0]) + (obj_loc[obj_index].y - results[1]))/2.0 + 0.04;
            target_point.point.z = obj_loc[obj_index].z;
            point_pub.publish(target_point);

            // tell manager object detect successfully
            id_data_msgs::ID_Data feedback;
            feedback.id = 3;
            feedback.data[0] = 15;
            result_pub.publish(feedback);
            free(boxes);
            DETECT_FLAG = 0;
        }
//         else if(car_adjust > 0) {
//            // tell arm where is object
//            id_data_msgs::ID_Data car_feedback;
//            car_feedback.id = 2;
//            car_feedback.data[0] = 6;
//            car_feedback.data[1] = 5;
//            result_pub.publish(car_feedback);
//        	DETECT_FLAG = 0;
//        } else if(car_adjust == 0) {
//            // tell manager object detect successfully
//            id_data_msgs::ID_Data feedback;
//            feedback.id = 3;
//            feedback.data[0] = 13;
//            result_pub.publish(feedback);
//            free(boxes);
//            DETECT_FLAG = 0;
//		}
	 	else {
            // tell manager object detect successfully
            id_data_msgs::ID_Data feedback;
            feedback.id = 3;
            feedback.data[0] = 13;
            result_pub.publish(feedback);
            free(boxes);
            DETECT_FLAG = 0;
		}

		printf("car_adjust=%d\n", car_adjust);
        cv::Mat ret = image_to_Mat(im);
        cv::imshow(DETECTION_WINDOW, ret);
        cv::waitKey(10);

        //free_image(im);
    }

    void callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                  const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
    {
        //printf("kinect callback\n");
        if(DETECT_FLAG == 1) ROS_INFO("callback detect msg\n");

        cv::Mat color, depth;

        readCameraInfo(cameraInfoColor, cameraMatrixColor);
        readCameraInfo(cameraInfoDepth, cameraMatrixDepth);
        //readImage(imageColor, color);
        readColorImage(imageColor, color);
        readImage(imageDepth, depth);

        // IR image input
        if(color.type() == CV_16U)
        {
            cv::Mat tmp;
            color.convertTo(tmp, CV_8U, 0.02);
            cv::cvtColor(tmp, color, CV_GRAY2BGR);
        }

        lock.lock();
        this->color = color;
        this->depth = depth;
        updateImage = true;
        updateCloud = true;
        lock.unlock();
    }



    void imageViewer()
    {
        cv::Mat color, depth, depthDisp, combined;
        //std::chrono::time_point<std::chrono::high_resolution_clock> start, now;
        //double fps = 0;
        //size_t frameCount = 0;
        //std::ostringstream oss;
        //const cv::Point pos(5, 15);
        //const cv::Scalar colorText = CV_RGB(255, 255, 255);
        //const double sizeText = 0.5;
        //const int lineText = 1;
        //const int font = cv::FONT_HERSHEY_SIMPLEX;

        //cv::namedWindow("Image Viewer");
        //oss << "starting...";

        //start = std::chrono::high_resolution_clock::now();
        for(; running && ros::ok();)
        {
            if(updateImage)
            {
                lock.lock();
                color = this->color;
                depth = this->depth;
                updateImage = false;
                lock.unlock();

                if(DETECT_FLAG==1) {
                    objectDetect(color, depth);
                }

                lock.lock();
                updateImage = false;
                lock.unlock();

                //        ++frameCount;
                //        now = std::chrono::high_resolution_clock::now();
                //        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0;
                //        if(elapsed >= 1.0)
                //        {
                //          fps = frameCount / elapsed;
                //          oss.str("");
                //          oss << "fps: " << fps << " ( " << elapsed / frameCount * 1000.0 << " ms)";
                //          start = now;
                //          frameCount = 0;
                //        }

                //        dispDepth(depth, depthDisp, 12000.0f);
                //        combine(color, depthDisp, combined);
                //combined = color;

                //cv::putText(combined, oss.str(), pos, font, sizeText, colorText, lineText, CV_AA);
                //std::cout<<"color type"<<color.type()<<std::endl;
                //cv::imshow(DETECTION_WINDOW, color);
                //cv::imshow("color", color);
            }

            //cv::waitKey(10);
            //int key = cv::waitKey(1);
            //      switch(key & 0xFF)
            //      {
            //      case 27:
            //      case 'q':
            //        running = false;
            //        break;
            //      case ' ':
            //      case 's':
            //        if(mode == IMAGE)
            //        {
            //          //createCloud(depth, color, cloud);
            //          //saveCloudAndImages(cloud, color, depth, depthDisp);
            //        }
            //        else
            //        {
            //          save = true;
            //        }
            //        break;
            //      }
        }
        //cv::destroyAllWindows();
        //cv::waitKey(100);
    }

    void cloudViewer()
    {
        cv::Mat color, depth;
        pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
        const std::string cloudName = "rendered";

        lock.lock();
        color = this->color;
        depth = this->depth;
        updateCloud = false;
        lock.unlock();

        createCloud(depth, color, cloud);

        visualizer->addPointCloud(cloud, cloudName);
        visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        visualizer->initCameraParameters();
        visualizer->setBackgroundColor(0, 0, 0);
        visualizer->setPosition(mode == BOTH ? color.cols : 0, 0);
        visualizer->setSize(color.cols, color.rows);
        visualizer->setShowFPS(true);
        visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
        visualizer->registerKeyboardCallback(&Receiver::keyboardEvent, *this);

        for(; running && ros::ok();)
        {
            if(updateCloud)
            {
                lock.lock();
                color = this->color;
                depth = this->depth;
                updateCloud = false;
                lock.unlock();

                createCloud(depth, color, cloud);

                visualizer->updatePointCloud(cloud, cloudName);
            }
            if(save)
            {
                save = false;
                cv::Mat depthDisp;
                dispDepth(depth, depthDisp, 12000.0f);
                saveCloudAndImages(cloud, color, depth, depthDisp);
            }
            visualizer->spinOnce(10);
        }
        visualizer->close();
    }

    void keyboardEvent(const pcl::visualization::KeyboardEvent &event, void *)
    {
        if(event.keyUp())
        {
            switch(event.getKeyCode())
            {
            case 27:
            case 'q':
                running = false;
                break;
            case ' ':
            case 's':
                save = true;
                break;
            }
        }
    }

    void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
    {
        cv_bridge::CvImageConstPtr pCvImage;
        pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
        //pCvImage = cv_bridge::toCvShare(msgImage, sensor_msgs::image_encodings::RGB8);
        pCvImage->image.copyTo(image);
    }

    void readColorImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
    {
        cv_bridge::CvImageConstPtr pCvImage;
        pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
        //std::cout<<"color img"<< msgImage->encoding<<std::endl;
        //pCvImage = cv_bridge::toCvShare(msgImage, sensor_msgs::image_encodings::RGB8);
        pCvImage->image.copyTo(image);
        cvtColor(image, image, CV_BGR2RGB);
    }

    void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const
    {
        double *itC = cameraMatrix.ptr<double>(0, 0);
        for(size_t i = 0; i < 9; ++i, ++itC)
        {
            *itC = cameraInfo->K[i];
        }
    }

    void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
    {
        cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
        const uint32_t maxInt = 255;

#pragma omp parallel for
        for(int r = 0; r < in.rows; ++r)
        {
            const uint16_t *itI = in.ptr<uint16_t>(r);
            uint8_t *itO = tmp.ptr<uint8_t>(r);

            for(int c = 0; c < in.cols; ++c, ++itI, ++itO)
            {
                *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
            }
        }

        cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
    }

    void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
    {
        out = cv::Mat(inC.rows, inC.cols, CV_8UC3);

#pragma omp parallel for
        for(int r = 0; r < inC.rows; ++r)
        {
            const cv::Vec3b
                    *itC = inC.ptr<cv::Vec3b>(r),
                    *itD = inD.ptr<cv::Vec3b>(r);
            cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

            for(int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO)
            {
                itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
                itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
                itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
            }
        }
    }



    void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) const
    {
        const float badPoint = std::numeric_limits<float>::quiet_NaN();

#pragma omp parallel for
        for(int r = 0; r < depth.rows; ++r)
        {
            pcl::PointXYZRGBA *itP = &cloud->points[r * depth.cols];
            const uint16_t *itD = depth.ptr<uint16_t>(r);
            const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
            const float y = lookupY.at<float>(0, r);
            const float *itX = lookupX.ptr<float>();

            for(size_t c = 0; c < (size_t)depth.cols; ++c, ++itP, ++itD, ++itC, ++itX)
            {
                register const float depthValue = *itD / 1000.0f;
                // Check for invalid measurements
                if(*itD == 0)
                {
                    // not valid
                    itP->x = itP->y = itP->z = badPoint;
                    itP->rgba = 0;
                    continue;
                }

                //convert kinect point to arm point
                cv::Mat kinect_point = cv::Mat::zeros(4, 1, CV_64FC1);
                kinect_point.at<double>(0,0) = *itX * depthValue;
                kinect_point.at<double>(1,0) = y * depthValue;
                kinect_point.at<double>(2,0) = depthValue;
                kinect_point.at<double>(3,0) = 1.0f;

                cv::Mat arm_point = cam2arm * kinect_point;

                Obj_loc o;
                o.x = arm_point.at<double>(0,0);
                o.y = arm_point.at<double>(1,0);
                o.z = arm_point.at<double>(2,0);

                itP->z = o.z;
                itP->x = o.x;
                itP->y = o.y;

                //        itP->z = depthValue;
                //        itP->x = *itX * depthValue;
                //        itP->y = y * depthValue;
                itP->b = itC->val[0];
                itP->g = itC->val[1];
                itP->r = itC->val[2];
                //        itP->r = itC->val[0];
                //        itP->g = itC->val[1];
                //        itP->b = itC->val[2];

                itP->a = 255;
            }
        }
    }

    void saveCloudAndImages(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depthColored)
    {
        oss.str("");
        oss << "./" << std::setfill('0') << std::setw(4) << frame;
        const std::string baseName = oss.str();
        const std::string cloudName = baseName + "_cloud.pcd";
        const std::string colorName = baseName + "_color.jpg";
        const std::string depthName = baseName + "_depth.png";
        const std::string depthColoredName = baseName + "_depth_colored.png";

        OUT_INFO("saving cloud: " << cloudName);
        writer.writeBinary(cloudName, *cloud);
        OUT_INFO("saving color: " << colorName);
        cv::imwrite(colorName, color, params);
        OUT_INFO("saving depth: " << depthName);
        cv::imwrite(depthName, depth, params);
        OUT_INFO("saving depth: " << depthColoredName);
        cv::imwrite(depthColoredName, depthColored, params);
        OUT_INFO("saving complete!");
        ++frame;
    }

    void createLookup(size_t width, size_t height)
    {
        const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
        const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
        const float cx = cameraMatrixColor.at<double>(0, 2);
        const float cy = cameraMatrixColor.at<double>(1, 2);
        float *it;

        //	printf("fx = %f, fy = %f, 1/fx=%f, 1/fy=%f\n", fx, fy, 1/fx, 1/fy);
        //	printf("cx = %f, cy = %f\n", cx, cy);

        lookupY = cv::Mat(1, height, CV_32F);
        it = lookupY.ptr<float>();
        for(size_t r = 0; r < height; ++r, ++it)
        {
            *it = (r - cy) * fy;
        }

        lookupX = cv::Mat(1, width, CV_32F);
        it = lookupX.ptr<float>();
        for(size_t c = 0; c < width; ++c, ++it)
        {
            *it = (c - cx) * fx;
        }
    }
};

void help(const std::string &path)
{
    std::cout << path << FG_BLUE " [options]" << std::endl
              << FG_GREEN "  name" NO_COLOR ": " FG_YELLOW "'any string'" NO_COLOR " equals to the kinect2_bridge topic base name" << std::endl
              << FG_GREEN "  mode" NO_COLOR ": " FG_YELLOW "'qhd'" NO_COLOR ", " FG_YELLOW "'hd'" NO_COLOR ", " FG_YELLOW "'sd'" NO_COLOR " or " FG_YELLOW "'ir'" << std::endl
              << FG_GREEN "  visualization" NO_COLOR ": " FG_YELLOW "'image'" NO_COLOR ", " FG_YELLOW "'cloud'" NO_COLOR " or " FG_YELLOW "'both'" << std::endl
              << FG_GREEN "  options" NO_COLOR ":" << std::endl
              << FG_YELLOW "    'compressed'" NO_COLOR " use compressed instead of raw topics" << std::endl
              << FG_YELLOW "    'approx'" NO_COLOR " use approximate time synchronization" << std::endl;
}





//void ugvReceiveCallback(const id_data_msgs::ID_Data::ConstPtr &msg)
//{
//	//printf("receive: msg id=%d\n", msg->id);
//	if (msg->id == 3 &&  msg->data[0]== 1) {
//		int flag = msg->data[0];

//		DETECT_FLAG = 1;

//		id_data_msgs::ID_Data feedback;
//		feedback.id = 3;
//		feedback.data[0] = 14;
//		result_pub.publish(feedback);
//	}

//}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "darknet_ros");

    //	thresh = find_float_arg(argc, argv, "-thresh", .10);
    //	hier_thresh = find_float_arg(argc, argv, "-hier", .5);

    thresh = .10;
    hier_thresh = .5;

    detector_init(datacfg, cfgfile, weightfile, &net, &names, &alphabet);
    printf("detector init OK\n");


    std::string ns = K2_DEFAULT_NS;
    std::string topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
    //std::string topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    std::string topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    bool useExact = false;
    bool useCompressed = false;
    Receiver::Mode mode = Receiver::IMAGE;
    //Receiver::Mode mode = Receiver::CLOUD;

    topicColor = "/" + ns + topicColor;
    topicDepth = "/" + ns + topicDepth;

    Receiver receiver(topicColor, topicDepth, useExact, useCompressed);
    printf("=========== wait detect cmd=================\n");

    receiver.run(mode);

    ros::shutdown();
    return 0;
}
