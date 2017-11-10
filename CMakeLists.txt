cmake_minimum_required(VERSION 2.8.3)
project(sia_darknet_ros)

set(GPU 0)
set(CUDNN 0)
set(OPENCV 1)
#set(DEBUG 0)

if(GPU)
set(ARCH
	-gencode arch=compute_20,code=[sm_20,sm_21]
	-gencode arch=compute_30,code=sm_30
	-gencode arch=compute_35,code=sm_35
	-gencode arch=compute_50,code=[sm_50,compute_50]
	-gencode arch=compute_52,code=[sm_52,compute_52]
	-gencode arch=compute_61,code=sm_61#added by hanker,2017102920:42 
	-gencode arch=compute_62,code=sm_62#added by hanker,2017102920:42
)
endif()

set(darknet_LIBRARIES m pthread stdc++)
set(darknet_INCLUDE_DIRS src/${PROJECT_NAME} src/${PROJECT_NAME}/darknet_src)
set(darknet_LINK_DIRS /usr/local/cuda/lib64)





if(GPU)
set(OPTS -O3)
else()
set(OPTS -Ofast)
endif()

#if(DEBUG) 
#	set(OPTS -O0 -g)
#endif()



if(OPENCV)
	add_definitions(-DOPENCV)
	set(darknet_INCLUDE_DIRS   ${darknet_INCLUDE_DIRS} /usr/include/opencv)
endif()

if(GPU)
	add_definitions(-DGPU)
	set(darknet_INCLUDE_DIRS   ${darknet_INCLUDE_DIRS} /usr/local/cuda/include)
	set(darknet_LIBRARIES ${darknet_LIBRARIES} cuda cudart cublas curand)
endif()

if(CUDNN)
	add_definitions(-DCUDNN)
	set(darknet_LIBRARIES ${darknet_LIBRARIES} cudnn)
endif()

set(CUDA_NVCC_FLAGS ${ARCH} ${OPTS} -Wno-deprecated-gpu-targets)
set(CMAKE_C_FLAGS ${OPTS})
set(CMAKE_CXX__FLAGS ${OPTS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBT_USE_DOUBLE_PRECISION -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wunused-function -Wunused-label -Wunused-variable -Wunused-but-set-variable -Wunused-but-set-parameter")
# Additional warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Warray-bounds -Wtype-limits -Wreturn-type -Wsequence-point -Wparentheses -Wmissing-braces -Wchar-subscripts -Wswitch -Wwrite-strings -Wenum-compare -Wempty-body -Wlogical-op")



# Check for c++11 support
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  MESSAGE(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()




set(OBJS src/darknet_ros.cpp	src/detector_ros.c
src/darknet_src/activation_layer.c       src/darknet_src/gru_layer.c
src/darknet_src/activations.c            src/darknet_src/im2col.c
src/darknet_src/art.c                    src/darknet_src/image.c
src/darknet_src/avgpool_layer.c          src/darknet_src/layer.c
src/darknet_src/batchnorm_layer.c        src/darknet_src/list.c
src/darknet_src/blas.c                   src/darknet_src/local_layer.c
src/darknet_src/box.c                    src/darknet_src/matrix.c
src/darknet_src/captcha.c                src/darknet_src/maxpool_layer.c
src/darknet_src/cifar.c                  src/darknet_src/network.c
src/darknet_src/classifier.c             src/darknet_src/nightmare.c
src/darknet_src/coco.c                   src/darknet_src/normalization_layer.c
src/darknet_src/col2im.c                 src/darknet_src/option_list.c
src/darknet_src/compare.c                src/darknet_src/parser.c
src/darknet_src/connected_layer.c        src/darknet_src/region_layer.c
src/darknet_src/convolutional_layer.c    src/darknet_src/reorg_layer.c
src/darknet_src/cost_layer.c             src/darknet_src/rnn.c
src/darknet_src/crnn_layer.c             src/darknet_src/rnn_layer.c
src/darknet_src/crop_layer.c             src/darknet_src/rnn_vid.c
src/darknet_src/cuda.c                   src/darknet_src/route_layer.c
src/darknet_src/darknet.c                src/darknet_src/shortcut_layer.c
src/darknet_src/data.c                   src/darknet_src/softmax_layer.c
src/darknet_src/deconvolutional_layer.c  src/darknet_src/super.c
src/darknet_src/demo.c                   src/darknet_src/swag.c
src/darknet_src/detection_layer.c        src/darknet_src/tag.c
src/darknet_src/detector.c               src/darknet_src/tree.c
src/darknet_src/dice.c                   src/darknet_src/utils.c
src/darknet_src/dropout_layer.c          src/darknet_src/voxel.c
src/darknet_src/gemm.c                   src/darknet_src/writing.c
src/darknet_src/go.c                     src/darknet_src/yolo.c
)

if(GPU)
set(OBJS ${OBJS} 
src/darknet_src/activation_kernels.cu
src/darknet_src/avgpool_layer_kernels.cu
src/darknet_src/blas_kernels.cu
src/darknet_src/col2im_kernels.cu
src/darknet_src/convolutional_kernels.cu
src/darknet_src/crop_layer_kernels.cu
src/darknet_src/deconvolutional_kernels.cu
src/darknet_src/dropout_layer_kernels.cu
src/darknet_src/im2col_kernels.cu
src/darknet_src/maxpool_layer_kernels.cu
src/darknet_src/network_kernels.cu
)
endif()

if(GPU)
find_package(CUDA REQUIRED)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(PCL REQUIRED)


if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()



## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  roscpp
  sensor_msgs
  std_msgs
  rostime message_filters compressed_image_transport compressed_depth_image_transport kinect2_bridge
  geometry_msgs
  pcl_conversions
  pcl_msgs
  pcl_ros
)

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   bbox.msg
   bbox_array.msg
 )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   sensor_msgs
   std_msgs
 )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES darknet_yolov2
#  CATKIN_DEPENDS cv_bridge image_transport message_generation roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${darknet_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
#  ${PCL_INCLUDE_DIRS}
#  ${kinect2_bridge_INCLUDE_DIRS}
   
)

link_directories(
  ${darknet_LINK_DIRS}
)

## Declare a C++ library
# add_library(darknet_yolov2
#   src/${PROJECT_NAME}/darknet_yolov2.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(darknet_yolov2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
#cuda_add_executable(darknet_ros_node ${OBJS})

if(GPU)
cuda_add_executable(darknet_ros_node_usb_cam ${OBJS})
target_link_libraries(darknet_ros_node_usb_cam
   ${catkin_LIBRARIES}
   ${darknet_LIBRARIES}
   ${OpenCV_LIBRARIES}
#  ${PCL_LIBRARIES}
#  ${kinect2_bridge_LIBRARIES}

   libvtkCommon.so libvtkFiltering.so
 )
else()
add_executable(darknet_ros_node ${OBJS})
target_link_libraries(darknet_ros_node
   ${catkin_LIBRARIES}
   ${darknet_LIBRARIES}
   ${OpenCV_LIBRARIES}
#  ${PCL_LIBRARIES}
#  ${kinect2_bridge_LIBRARIES}
   libvtkCommon.so libvtkFiltering.so
 )
endif()

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(darknet_yolov2_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
 

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS darknet_yolov2 darknet_yolov2_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_darknet_yolov2.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
