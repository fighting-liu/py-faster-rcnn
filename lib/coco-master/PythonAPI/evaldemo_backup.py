#!/usr/bin python
# -*- coding: utf-8 -*-
# step1
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# set before running
choice = {'summarize':1,'filter':0,'print':0}
dataset = {'all':1, 'part':0}

# step2
annType = 'bbox'
print 'Running demo for *%s* results.'%(annType)

# step3
# annFile = "../data/annotations/instances_val2014_bak.json"
annFile = "/data/home/liuhuawei/hanbag1000_and_clothing3000_test.json"
# annFile = '/data/home/liuhuawei/clothing_test_sample.json'
# annFile = "2500_ann_samson.json"
# Get dataset,anns,cats,imgs,imgToAnns,catToImgs
cocoGt=COCO(annFile)

# step4
# resFile = "2500_prop_samson.json"
# resFile = "2500_prop_samson.json"
resFile = "/home/liuhuawei/detection_hangbag1000_and_clothing3000_vgg.json"
# resFile = '/data/home/liuhuawei/detection_clothing_test_sample.json'
# resFile = "detection.json"
# cocoDt.dataset['images'] 同 cocoGt.dataset['images']
# category同上,ie,cocoDt与同cocoGt相同
# anns即整个precomputed proposal
cocoDt=cocoGt.loadRes(resFile)

# step5
imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:10]
print "imgids",imgIds
catIds = sorted(cocoGt.getCatIds())
print "CatIds:", catIds

# step6
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.params.useCats = 0
if dataset['part'] == 1:
    import json
    f = file(resFile)
    js = json.load(f)
    images = [i['image_id'] for i in js]
    # cocoEval.params.imgIds = sorted(set([19934,177327,60471,148964,97381]))
    cocoEval.params.imgIds = sorted(set(images))
    cocoEval.params.useCats = 1
if choice['print'] == 1:
    cocoEval.params.maxDets = [1,10,100,200,300,400,500,600,700,800,900,1000]

cocoEval.evaluate()

if choice['filter'] == 1:
    cocoEval.calarap()
    cocoEval.filter_image(maxDets=1000)
elif choice['print'] == 1:
    cocoEval.accumulate()
    cocoEval.summarize(print_curve='yes')
elif choice['summarize'] == 1:
    cocoEval.accumulate()
    cocoEval.summarize()
