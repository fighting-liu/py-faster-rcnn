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

import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('pdf')
import matplotlib.pyplot as plt

CLASSES = ('__background__', 'upbody', 'downbody', 'fullbody')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    print inds
    if len(inds) == 0:
        # cv2.imshow('test', im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
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

# def demo(net, image_name):
#     """Detect object classes in an image using pre-computed object proposals."""

#     # Load the demo image
#     im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
#     im = cv2.imread(im_file)

#     # Detect all object classes and regress object bounds
#     timer = Timer()
#     timer.tic()
#     scores, boxes = im_detect(net, im)
#     timer.toc()
#     print ('Detection took {:.3f}s for '
#            '{:d} object proposals').format(timer.total_time, boxes.shape[0])

#     # Visualize detections for each class
#     CONF_THRESH = 0.8
#     NMS_THRESH = 0.3
#     for cls_ind, cls in enumerate(CLASSES[1:]):
#         cls_ind += 1 # because we skipped background
#         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#         cls_scores = scores[:, cls_ind]
#         dets = np.hstack((cls_boxes,
#                           cls_scores[:, np.newaxis])).astype(np.float32)
#         keep = nms(dets, NMS_THRESH)
#         dets = dets[keep, :]
#         vis_detections(im, cls, dets, thresh=CONF_THRESH)

## absolute path input for image names
def bag_demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(image_name)
    print im.shape
    # Detect all object classes and regress object bounds
    # timer = Timer()
    # timer.tic()
    # scores, boxes = im_detect(net, im)
    # timer.toc()
    # print ('Detection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # # Visualize detections for each class
    # CONF_THRESH = 0.5
    # NMS_THRESH = 0.3
    # for cls_ind, cls in enumerate(CLASSES[1:]):
    #     cls_ind += 1 # because we skipped background
    #     cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #     cls_scores = scores[:, cls_ind]
    #     dets = np.hstack((cls_boxes,
    #                       cls_scores[:, np.newaxis])).astype(np.float32)
    #     keep = nms(dets, NMS_THRESH)
    #     dets = dets[keep, :]
    #     vis_detections(im, cls, dets, thresh=CONF_THRESH)

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

def index_to_fullPath_dict(filename):
        dic = {}
        with open(filename, 'rb') as f:
            for line in f.readlines():
                words = line.split(' ')
                full_path = words[0]
                index = full_path.split('/')[-1]
                dic[index.split('.')[0]] = full_path
        return dic

def sample_test_images(annot_path, test_path, sample_num=10):
    ## return the sampled full path list
    sampled_path = []
    index_to_path_dict = index_to_fullPath_dict(annot_path)
    test_index = []
    with open(test_path, 'rb') as f:
        for line in f.readlines():
            if line[-1] == '\n':
                test_index.append(line[:-1])
            else:
                test_index.append(line)
    if sample_num > len(test_index):
        print "Too many samples for testing!!!"
        return                            
    permu = np.random.permutation(len(test_index))
    sampled_index = np.array(test_index)[permu[:sample_num]].tolist()
    sampled_path = [index_to_path_dict[index] for index in sampled_index]  
    return sampled_path

def load_images(test_path): 
    test_index = []
    with open(test_path, 'rb') as f:
        for line in f.readlines():
            test_index.append(line.strip(' \n'))
            # if line[-1] == '\n':
            #     test_index.append(line[:-1])
            # else:
            #     test_index.append(line) 
    return test_index            

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    # caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                           NETS[args.demo_net][1])

    # prototxt = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test_backup.prototxt'
    # caffemodel = '/data/home/liuhuawei/detection/py-faster-rcnn/output/faster_rcnn_end2end/backup/vgg16_faster_rcnn_iter_30000.caffemodel'
    prototxt = 'models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = 'output/faster_rcnn_end2end/voc_2007_trainval/clothing/vgg16_faster_rcnn_iter_200000.caffemodel'    
    # prototxt = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test_with_only_foveal.prototxt'
    # caffemodel = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/only_foveal/vgg16_faster_rcnn_iter_60000.caffemodel'    

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

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    test_path = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test_tmp.txt'
    # im_names = sample_test_images(annot_path, test_path, sample_num=10)
    im_names = load_images(test_path)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        bag_demo(net, '/data/home/liuhuawei/clothing_data/Img/'+im_name)
    # ### This should be images of absolute path.    
    # # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    # #             '001763.jpg', '004545.jpg']
    # # annot_path = '/home/liuhuawei/detection/annotate_data.txt'
    # test_path = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test_tmp.txt'
    # # im_names = sample_test_images(annot_path, test_path, sample_num=10)
    # im_names = load_images(test_path)
    # for im_name in im_names:
    #     print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #     print 'Demo for data/demo/{}'.format(im_name)
    #     bag_demo(net, '/data/home/liuhuawei/'+im_name)

    plt.show()
