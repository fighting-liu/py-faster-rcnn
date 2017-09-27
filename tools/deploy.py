# -*- coding: utf-8 -*
#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json
import logging

import time

# from data_augmentation_back import data_augmentationc

# CLASSES = ('__background__', 'bag')
CLASSES = ('__background__', 'upbody', 'downbody', 'fullbody')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='clothing_detection_process_task_taobao_4.log',
                filemode='w')
logging.info('This is info message')

## absolute path input for image names
def bag_demo(net, im_name, image_id, index):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    # bbox = []
    # area = 0
    # iscrowd = 0
    # category_id = -1

    im_name = im_name if im_name[:5] == '/core' else '/core'+im_name
    im = cv2.imread(im_name)
    # if min(im.shape[:2]) < 100:
    #     return    
    # im = cv2.imread('/core'+image_name)
    H, W, _ = im.shape

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3

    #########
    bbox = []
    category_id = -1
    score_max = 0
    #########    

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]      
        if len(inds) > 0:
            ind = inds[0]
            if dets[ind, -1] > score_max:
                category_id = cls_ind
                score_max = dets[ind, -1]
                bbox = [int(dets[ind, 0]), int(dets[ind, 1]), int(dets[ind, 2])-int(dets[ind, 0])+1, int(dets[ind, 3])-int(dets[ind, 1])+1]
    print score_max, bbox, category_id                  
    return {'id': index, 'image_id': image_id, 'bbox': bbox,'category_id': category_id}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def load_json(json_file):
    assert os.path.exists(json_file), \
            'json file not found at: {}'.format(json_file)
    with open(json_file, 'rb') as f:
        data = json.load(f)     
    return data        


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    cfg_file = '/core/data/deploy/server_image/faster_rcnn_end2end_clothing.yml'


    prototxt = '/core/data/deploy/server_image/test_clothing.prototxt'
    caffemodel = '/core/data/deploy/server_image/vgg16_clothing_240000.caffemodel'
   # prototxt = 'models/handbag_and_clothing/VGG16/faster_rcnn_end2end/test.prototxt'
    #caffemodel = 'output/faster_rcnn_end2end/handbag_and_clothing_2007_trainval/vgg16_faster_rcnn_v2_iter_70000.caffemodel'      
    input_json_file = '/core1/data/home/liuhuawei/input/clothing_to_detect_task_4.json'
    # output_json_file = '/data/home/liuhuawei/11w_output.json'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if cfg_file is not None:
        cfg_from_file(cfg_file) 

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
 
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    annotations_list = []
    images_list = []
    data = load_json(input_json_file)
    name_index = 1
    for index, item in enumerate(data["images"]):
        index += 1
        # print "Processing %d/%d!!!" % (index, len(data["images"]))
        logging.info("Processing %d/%d!!!" % (index, len(data["images"])))
        images_list.append(item)
        image_name = item['file_name']
        image_id = item['id']
        s_time = time.time()
        try:
            annotations_list.append(bag_demo(net, image_name, image_id, index))
        except Exception, e:
            print e
            pass
        # print "Pcocessing time is %.3fs/img" % (time.time() - s_time, )  
        logging.info('gross time is: %f' % (time.time()-s_time))       

        if index % 10000 == 0 or index == len(data["images"]):
            output_dict = {"images": images_list, "type": data["type"], "annotations": annotations_list,\
                "categories": data["categories"]}  
            with open('/core1/data/home/liuhuawei/output/wtable0622_deepfashion/wtable_task4_%dw_output.json' % (name_index), 'wt') as f:
                f.write(json.dumps(output_dict))                 
            name_index += 1
            annotations_list = []
            images_list = []    


    






