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

# CLASSES = ('__background__', 'bag')
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

person_dir = "/data/home/liuhuawei/detection/dev-py-faster-rcnn/test_model/person"
# recall_dir = "/data/home/liuhuawei/detection/dev-py-faster-rcnn/test_model/recall"
# precision_dir = "/data/home/liuhuawei/detection/dev-py-faster-rcnn/test_model/precision"
# empty_dir = "/data/home/liuhuawei/detection/dev-py-faster-rcnn/test_model/empty"

def vis_detections(im, class_name, dets, gt_bbox, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    ax.add_patch(
        plt.Rectangle((gt_bbox[0], gt_bbox[1]),
                  gt_bbox[2] - gt_bbox[0],
                  gt_bbox[3] - gt_bbox[1], fill=False,
                  edgecolor='blue', linewidth=3.5)
    )
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

## absolute path input for image names
def bag_demo(net, image_name, bb, ovthresh, index, person_index=15):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background

        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[inds]

        if cls_ind != person_index:
            continue
        else:
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            vis_detections(im, cls, dets, bb, thresh=CONF_THRESH)   
            plt.savefig(os.path.join(person_dir, '%d.jpg' % index))          

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

def find_gt_bbox(annotations, image_index):
    for anno in annotations:
        if anno['source_id'] == image_index:
            return np.array(anno['bbox'])       


def imgs_in_path(path):
    if os.path.exists(path):
        print "Read images at dir: {:s}".format(path)
    else:
        print "{:s} not found!!".format(path)
        return None

    return glob.glob(os.path.join(path, '*.jpg'))


            

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id


    prototxt = 'models/pascal_voc/VGG16/faster_rcnn_end2end/demo.prototxt'
    caffemodel = 'data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'
    image_json_file = '/home/liuhuawei/7000_ann_from8650.json'    

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

    for p1 in path_list_1:
        for p2 in path_list_2:
            for p3 in path_list_3:
                path = os.path.join(base_path, p1, p2, p3)
                images = imgs_in_path(path)
                print images



    # data = load_json(image_json_file)
    # images = data["images"]
    # annotations = data["annotations"]

    # for index, item in enumerate(images[:4000]):
    #     print "Processing %d/%d!!!" % (index, len(data["images"]))

    #     image_name = item['file_name']
    #     image_index = item['img']

    #     gt_bbox = find_gt_bbox(annotations, image_index)
    #     gt_bbox = np.array([gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2]-1,\
    #         gt_bbox[3]+gt_bbox[1]-1])
    #     ##iou is 0.5
    #     bag_demo(net, image_name, gt_bbox, 0.5, index)

