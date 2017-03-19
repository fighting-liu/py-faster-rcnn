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
matplotlib.use('Agg')
# matplotlib.use('pdf')
import matplotlib.pyplot as plt

CLASSES = ('__background__', 'handbags', 'shoes', 'up_body', 'down_body', \
         'all_body', 'boots', 'bra', 'underwear', 'skirt', 'dress', 'makeup')

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    import numpy as np
    import urllib
    import cv2
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image    

def bag_demo(net, image_name, cat_ids, bboxes):
    
    """Detect object classes in an image using pre-computed object proposals."""
    im = cv2.imread(image_name)
    # im = url_to_image(image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    # print scores
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    colors = plt.cm.hsv(np.linspace(0, 1, len(CLASSES))).tolist()
    plt.imshow(im)
    currentAxis = plt.gca()    
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
        keep_final = np.where(dets[:, 4]>CONF_THRESH)[0]
        for i in keep_final:
            xmin = dets[i, 0]
            ymin = dets[i, 1]
            xmax = dets[i, 2]
            ymax = dets[i, 3]
            score = dets[i, 4]
            label_name = cls
            display_txt = '%s: %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[cls_ind]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})      
    plt.show() 

def bag_demo_double(net, image_name, cat_ids, bboxes):
    
    """Detect object classes in an image using pre-computed object proposals."""
    im = cv2.imread(image_name)
    # im = url_to_image(image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    # print scores
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    #######
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    #######
    colors = plt.cm.hsv(np.linspace(0, 1, len(CLASSES))).tolist()
    ax1.imshow(im)
    currentAxis = plt.gca()    
    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)

        dets = dets[keep, :]
        keep_final = np.where(dets[:, 4]>CONF_THRESH)[0]
        for i in keep_final:
            xmin = dets[i, 0]
            ymin = dets[i, 1]
            xmax = dets[i, 2]
            ymax = dets[i, 3]
            score = dets[i, 4]
            label_name = cls
            display_txt = '%s: %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[cls_ind]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(im)
    currentAxis = plt.gca()  
    for idx, cls_ind in enumerate(cat_ids):
        cls = CLASSES[cls_ind]
        xmin = bboxes[idx, 0]
        ymin = bboxes[idx, 1]
        xmax = xmin + bboxes[idx, 2] - 1
        ymax = ymin + bboxes[idx, 3] - 1
        label_name = cls
        display_txt = '%s: %.2f'%(label_name, 1)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[cls_ind]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})     
    plt.show()     

def bag_demo_origin(image_name, cat_ids, bboxes):   
    im = cv2.imread(image_name)
    im = im[:, :, (2, 1, 0)]
    colors = plt.cm.hsv(np.linspace(0, 1, len(CLASSES))).tolist()
    plt.imshow(im)
    currentAxis = plt.gca()   

    for idx, cls_ind in enumerate(cat_ids):
        cls = CLASSES[cls_ind]
        xmin = bboxes[idx, 0]
        ymin = bboxes[idx, 1]
        xmax = xmin + bboxes[idx, 2] - 1
        ymax = ymin + bboxes[idx, 3] - 1
        label_name = cls
        display_txt = '%s: %.2f'%(label_name, 1)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[cls_ind]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})     
    plt.show()    
    plt.draw()                         


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    # parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
    #                     choices=NETS.keys(), default='vgg16')

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
            if line[-1] == '\n':
                test_index.append(line[:-1])
            else:
                test_index.append(line) 
    return test_index            

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test_backup.prototxt'
    # caffemodel = '/data/home/liuhuawei/detection/py-faster-rcnn/output/faster_rcnn_end2end/backup/vgg16_faster_rcnn_iter_30000.caffemodel'
    prototxt = 'models/all_categories_0307/VGG16/faster_rcnn_end2end/test.prototxt'
    # prototxt = 'models/all_categories/ResNet101/faster_rcnn_end2end/test.prototxt'
    # caffemodel = 'output/faster_rcnn_end2end/2007_trainval_all_categories/ResNet101_all_categories_frcnn_iter_100000.caffemodel'
    caffemodel = 'output/faster_rcnn_end2end/2007_trainval_all_categories_0307/VGG16_all_categories_0307_frcnn_iter_300000.caffemodel'    
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
    # print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    #####################    
    import json
    json_file = '/data/home/liuhuawei/detection/input/clothing_all_cats_test_0307.json'
    with open(json_file, 'rb') as f:
        data = json.load(f)
    images = data['images']
    all_image_index = [img_dict['img'].split('.')[0] for img_dict in images]  
    distinct_image_index = set(all_image_index)   
    ## 2. make index to abspath dict for indexing       
    img_idx_abspath = [(img_dict['img'].split('.')[0], img_dict['file_name']) for img_dict in images]
    _index_to_path = dict(img_idx_abspath)   
    ## 3
    annotations = data['annotations']
    combined_annts = {}
    for annot in annotations:
        image_index = annot['source_id'].split('.')[0]
        if image_index not in combined_annts.keys():
            combined_annts[image_index] = []
        combined_annts[image_index].append(annot) 
    ## 4    
    for idx, img_index in enumerate(distinct_image_index):
        print "Converting json to xml %d/%d" % (idx+1, len(distinct_image_index))
        annots = combined_annts[img_index]
        img_ids = [an['image_id'] for an in annots]
        assert len(set(img_ids)) == 1          

        cat_ids = [an['category_id'] for an in annots]
        bboxes = np.array([an['bbox'] for an in annots])
        assert bboxes.shape[0] == len(img_ids)

        abspath = _index_to_path[img_index]
        bag_demo_double(net, abspath, cat_ids, bboxes)
        # bag_demo(net, abspath)
        # bag_demo_origin(abspath, cat_ids, bboxes)        

    # file_names = [img['file_name'] for img in images]
    # im_names = set(file_names)
    # print "Num of test images are {}".format(len(im_names))
    # print images[0] 
    # all_image_index = [img_dict['img'].split('.')[0] for img_dict in images]  
    #     distinct_image_index = set(all_image_index)        
    # im_names = ['http://ci.xiaohongshu.com/21a63b7e-8bb6-468e-8252-3b9d26b8fc66@r_1280w_1280h.jpg']
    # for im_name in im_names:
    #     print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #     print 'Demo for data/demo/{}'.format(im_name)
    #     bag_demo(net, im_name)
    #     bag_demo_origin(im_name, )
