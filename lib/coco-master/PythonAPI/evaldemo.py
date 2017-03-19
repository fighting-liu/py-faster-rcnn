#!/usr/bin python
# -*- coding: utf-8 -*-
# step1
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import argparse
import sys
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Eveluate detection performance!')
    parser.add_argument('--annFile', dest='annFile',
                        help='annotation file',
                        default=None, type=str)
    parser.add_argument('--resFile', dest='resFile',
                    help='detection result file',
                    default=None, type=str)
    parser.add_argument('--class_names', dest='class_names',
                help='name of classes',
                default=None, type=str)
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse_args()

annFile = args.annFile
resFile = args.resFile
# cocoGt=COCO(annFile)
# cocoDt=cocoGt.loadRes(resFile)
annType = 'bbox'
print 'Running demo for *%s* results.'%(annType)

choice = {'summarize':1,'filter':0,'print':0}
class_names = args.class_names.split(',')
if class_names is None:
    num_class = 2
else:
    num_class = len(class_names)    
if num_class > 2:
    for i in xrange(1, num_class):
        cocoGt=COCO(annFile)
        cocoDt=cocoGt.loadRes(resFile)
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.maxDets = [1, 10, 100] 
        cocoEval.params.useCats = 1
        cocoEval.params.catIds = [i] 
        cocoEval.evaluate()
        print 'Evaluation results for class {}!'.format(class_names[i])
        if choice['filter'] == 1:
            cocoEval.calarap()
            cocoEval.filter_image(maxDets=1000)
        elif choice['print'] == 1:
            cocoEval.accumulate()
            cocoEval.summarize(print_curve='yes')
        elif choice['summarize'] == 1:
            cocoEval.accumulate()
            cocoEval.summarize() 
cocoGt=COCO(annFile)
cocoDt=cocoGt.loadRes(resFile)            
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.maxDets = [1, 10, 100] 
cocoEval.params.useCats = 1
cocoEval.evaluate()
print 'Evaluation results all class!'
if choice['filter'] == 1:
    cocoEval.calarap()
    cocoEval.filter_image(maxDets=1000)
elif choice['print'] == 1:
    cocoEval.accumulate()
    cocoEval.summarize(print_curve='yes')
elif choice['summarize'] == 1:
    cocoEval.accumulate()
    cocoEval.summarize()             






# set before running

# dataset = {'all':0, 'part':1}
# use_cat = True
# if use_cat:
#     cocoEval.params.useCats = 1
    



      


# # step2

# print 'Running demo for *%s* results.'%(annType)

# # step3
# annFile = "/data/home/liuhuawei/hanbag1000_and_clothing3000_test.json"


# # step4
# resFile = "/home/liuhuawei/detection_hangbag1000_and_clothing3000_vgg.json"


# cocoEval = COCOeval(cocoGt,cocoDt,annType)
# if use_cat:
#     cocoEval.params.useCats = 1
#     cocoEval.params.maxDets = [10, 100, 300]     
#     # cocoEval.params.catIds = []

# if choice['print'] == 1:
#     cocoEval.params.maxDets = [1,10,100,200,300,400,500,600,700,800,900,1000]

# cocoEval.evaluate()

# if choice['filter'] == 1:
#     cocoEval.calarap()
#     cocoEval.filter_image(maxDets=1000)
# elif choice['print'] == 1:
#     cocoEval.accumulate()
#     cocoEval.summarize(print_curve='yes')
# elif choice['summarize'] == 1:
#     cocoEval.accumulate()
#     cocoEval.summarize()    


# # step5
# imgIds=sorted(cocoGt.getImgIds())
# catIds = sorted(cocoGt.getCatIds())

# step6 
## we will initialize params here 

## correct the initialized parameters
# cocoEval.params.imgIds  = imgIds
# cocoEval.params.useCats = 1
# if dataset['part'] == 1:
#     import json
#     f = file(resFile)
#     js = json.load(f)
#     images = [i['image_id'] for i in js]
#     # cocoEval.params.imgIds = sorted(set([19934,177327,60471,148964,97381]))
#     cocoEval.params.imgIds = sorted(set(images))
#     cocoEval.params.useCats = 1
#     cocoEval.params.maxDets = [10, 100, 300]   
#     # cocoEval.params.catIds = [4]

