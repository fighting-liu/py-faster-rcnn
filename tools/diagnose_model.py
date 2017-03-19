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
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json

CLASSES = ('__background__', 'bag')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
recall_dir = "/data/home/liuhuawei/detection/dev-py-faster-rcnn/test_model/res_101_update_bn_iter_65000/recall"
precision_dir = "/data/home/liuhuawei/detection/dev-py-faster-rcnn/test_model/res_101_update_bn_iter_65000/precision"
empty_dir = "/data/home/liuhuawei/detection/dev-py-faster-rcnn/test_model/res_101_update_bn_iter_65000/empty"

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
def bag_demo(net, image_name, bb, ovthresh, index, img_name):
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

        if dets.shape[0] != 0:
            ixmin = np.maximum(dets[:, 0], bb[0])
            iymin = np.maximum(dets[:, 1], bb[1])
            ixmax = np.minimum(dets[:, 2], bb[2])
            iymax = np.minimum(dets[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (dets[:, 2] - dets[:, 0] + 1.) *
                   (dets[:, 3] - dets[:, 1] + 1.) - inters)

            overlaps = inters / uni
            up_inds = np.where(overlaps >= ovthresh)[0]
            down_inds = np.where(overlaps < ovthresh)[0]

            if len(up_inds) == 0:
                vis_detections(im, cls, dets, bb, thresh=0.05)
                plt.savefig(os.path.join(recall_dir, '%s.jpg' % img_name))
                return                

            if len(down_inds) != 0:
                vis_detections(im, cls, dets, bb, thresh=0.05)
                plt.savefig(os.path.join(precision_dir, '%s.jpg' % img_name))             
                # plt.show()
        else:  
            cv2.imwrite(os.path.join(empty_dir, '%s.jpg' % img_name), im)            
                      


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

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()


    # prototxt = '/data/home/liuhuawei/detection/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
    # caffemodel = '/data/home/liuhuawei/detection/py-faster-rcnn/output/faster_rcnn_end2end/backup/vgg16_faster_rcnn_iter_30000.caffemodel'
    prototxt = '/data/home/liuhuawei/detection/py-faster-rcnn/models/pascal_voc/ResNet101/faster_rcnn_end2end/test.prototxt'
    caffemodel = '/data/home/liuhuawei/detection/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/resnet101_bag__iter_65000.caffemodel'    

    image_json_file = '/home/liuhuawei/1000_ann_from8650.json'
    # output_json_file = '/data/home/liuhuawei/11w_output.json'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    data = load_json(image_json_file)
    images = data["images"]
    annotations = data["annotations"]

    for index, item in enumerate(images):
        print "Processing %d/%d!!!" % (index, len(data["images"]))

        image_name = item['file_name']
        image_index = item['img']

        gt_bbox = find_gt_bbox(annotations, image_index)
        gt_bbox = np.array([gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2]-1,\
            gt_bbox[3]+gt_bbox[1]-1])
        ##iou is 0.5
        bag_demo(net, image_name, gt_bbox, 0.5, index, img_name=image_index)



    






