#include "darknet_src/network.h"
#include "darknet_src/region_layer.h"
#include "darknet_src/cost_layer.h"
//#include "darknet_src/utils.h"
#include "darknet_src/parser.h"
#include "darknet_src/box.h"
#include "darknet_src/demo.h"
#include "darknet_src/option_list.h"

#include "darknet_ros.hpp"
//#include <darknet_ros/bbox_array.h>
//#include <darknet_ros/bbox.h>

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


/*ROS_box* get_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)*/
/*{*/
/*    int i;*/

/*    for(i = 0; i < num; ++i){*/
/*        int class = max_index(probs[i], classes);*/
/*        float prob = probs[i][class];*/
/*        if(prob > thresh){*/

/*            int width = im.h * .012;*/

/*            if(0){*/
/*                width = pow(prob, 1./2.)*10+1;*/
/*                alphabet = 0;*/
/*            }*/

/*            printf("%s: %.0f%%\n", names[class], prob*100);*/
/*            int offset = class*123457 % classes;*/
/*            float red = get_color(2,offset,classes);*/
/*            float green = get_color(1,offset,classes);*/
/*            float blue = get_color(0,offset,classes);*/
/*            float rgb[3];*/

/*            //width = prob*20+2;*/

/*            rgb[0] = red;*/
/*            rgb[1] = green;*/
/*            rgb[2] = blue;*/
/*            box b = boxes[i];*/

/*            int left  = (b.x-b.w/2.)*im.w;*/
/*            int right = (b.x+b.w/2.)*im.w;*/
/*            int top   = (b.y-b.h/2.)*im.h;*/
/*            int bot   = (b.y+b.h/2.)*im.h;*/

/*            if(left < 0) left = 0;*/
/*            if(right > im.w-1) right = im.w-1;*/
/*            if(top < 0) top = 0;*/
/*            if(bot > im.h-1) bot = im.h-1;*/

/*            draw_box_width(im, left, top, right, bot, width, red, green, blue);*/
/*            if (alphabet) {*/
/*                image label = get_label(alphabet, names[class], (im.h*.03)/10);*/
/*                draw_label(im, top + width, left, label, rgb);*/
/*            }*/
/*        }*/
/*    }*/
/*}*/



int detector_init(char *datacfg, char *cfgfile, char *weightfile, network* net, char ***names, image ***alphabet)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    *names = get_labels(name_list);

    *alphabet = load_alphabet();
    *net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(net, weightfile);
    }
    set_batch_network(net, 1);
    srand(2222222);

	return net;

}


ROS_box* test_detector_ros(network net, image im, char **names, image **alphabet, float thresh, float hier_thresh)
{
    clock_t time;
    int i,j;
    float nms=.02;

	save_image(im, "input_color_image");
        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];
	//printf("im width=%d, height=%d\n", im.w, im.h);

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        //printf("Predicted in %f seconds.\n", sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
        if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

        save_image(im, "predictions");
        //show_image(im, "predictions");

/*	printf("probs======\n");*/

/*	for(i=0; i<l.w*l.h*l.n; i++) {*/
/*		for(j=0; j<l.classes; j++) {*/
/*			if(probs[i][j])*/
/*				printf("i,j:%d,%d, %f ", i,j, probs[i][j]);*/
/*		}*/
/*	}*/


	// extract the bounding boxes and send them to ROS
	ROS_box* ROI_boxes = (ROS_box *)calloc(l.w*l.h*l.n, sizeof(ROS_box));

	int count = 0;
	for(i = 0; i < l.w*l.h*l.n; ++i){

        int class = max_index(probs[i], l.classes);
        float prob = probs[i][class];

		if(prob > thresh) {

			box b = boxes[i];
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

/*			printf("class:%d\n", class);*/
/*			printf("left:%d\n", left);*/
/*			printf("right:%d\n", right);*/
/*			printf("top:%d\n", top);*/
/*			printf("bot:%d\n", bot);*/


			 ROI_boxes[count].x = (left+right)/2;
		     ROI_boxes[count].y = (top+bot)/2;
		     ROI_boxes[count].w = right-left;
		     ROI_boxes[count].h = bot-top;
		     ROI_boxes[count].Class = class;
			 count++;

			// define bounding box 
			// bbox must be 1% size of frame (3.2x2.4 pixels)
/*			if (bbox_width > 0.01 && bbox_height > 0.01) {*/
/*	   		     ROI_boxes[count].x = x_center;*/
/*		         ROI_boxes[count].y = y_center;*/
/*		         ROI_boxes[count].w = bbox_width;*/
/*		         ROI_boxes[count].h = bbox_height;*/
/*		         ROI_boxes[count].Class = class;*/

/*			printf("box#%d\n", count);*/
/*			printf("Class:%d\n", ROI_boxes[count].Class);*/
/*			printf("x:%d\n", ROI_boxes[count].x);*/
/*			printf("y:%d\n", ROI_boxes[count].y);*/
/*			printf("w:%d\n", ROI_boxes[count].w);*/
/*			printf("h:%d\n", ROI_boxes[count].h);*/

/*				 count++;*/
/*			}*/
		}
	}
	
	// create array to store found bounding boxes
	// if no object detected, make sure that ROS knows that num = 0
	if (count == 0) {
	    ROI_boxes[0].num = 0;
	} else {
	    ROI_boxes[0].num = count;
	}

	//printf("ROI boxes num:%d\n",count);


        //free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
        //cvWaitKey(1);
        //cvDestroyAllWindows();
#endif
/*        if (filename) break;*/
/*    }*/

	return ROI_boxes;
}
