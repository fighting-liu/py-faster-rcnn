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
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json

import glob

import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('pdf')
import matplotlib.pyplot as plt

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


## absolute path input for image names
def is_person(net, image_name, person_index=15):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(image_name)
    if im is None:
        return False

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)

    # Visualize detections for each class
    CONF_THRESH = 0.95
    NMS_THRESH = 0.3

    ##index for person
    cls_ind = 15

    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    dets = dets[inds]
    if len(inds) != 0:
        return True
    return False            

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
 


def imgs_in_path(path):
    if os.path.exists(path):
        print "Read images at dir: {:s}".format(path)
    else:
        print "{:s} not found!!".format(path)
        return None

    return glob.glob(os.path.join(path, '*.jpg'))

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
            

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id


    prototxt = 'models/pascal_voc/VGG16/faster_rcnn_end2end/demo.prototxt'
    caffemodel = 'data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'
    
    person_txt_file_path = 'data/person.txt' 

    path_list_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                 'a', 'b', 'c', 'd', 'e', 'f',]
    path_list_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                 'a', 'b', 'c', 'd', 'e', 'f',]
    path_list_3 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                 'a', 'b', 'c', 'd', 'e', 'f',]                                  
    base_path = '/data/xiaoshongshu/img/H'



    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if not os.path.isfile(prototxt):
        raise IOError(('{:s} not found.\n').format(prototxt))       
        

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)


    num_imgs = 100000
    counter = 0
    with open(person_txt_file_path, 'w') as file:
        for p1 in path_list_1:
            for p2 in path_list_2:
                for p3 in path_list_3:
                    path = os.path.join(base_path, p1, p2, p3)
                    images = imgs_in_path(path)

                    if images is not None:
                        for img in images:
                            if is_person(net, img, person_index=15):
                                file.write(img+'\n')
                    counter += len(images)                                
                    if counter >= num_imgs:
                        raise IOError("We finish the task!") 
                    else:
                        print "Processing %d/%d" % (counter, num_imgs)    



