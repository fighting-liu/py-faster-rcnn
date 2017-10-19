# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.all import AllCats
from datasets.coco_all import coco
import numpy as np

def get_imdb(name):
    """Get an imdb (image database) by name.
       previous name format: 2007_trainval(test)_taskname
       now name format: voc(coco)_2007_trainval(test)_taskname
    """
    ## e.g coco_2007_trainval_newclothing0802 
    data_type, year, split, task_name = name.split('_', 3)
    print data_type, task_name, split, year
    assert data_type in ['coco', 'voc']
    assert split in ['trainval', 'test'] and year == '2007'
    test_flag = True if split == 'test' else False
    if data_type == 'voc':
        return AllCats(task_name=task_name, image_set=split, year=year, test=test_flag)
    else:
        return coco(task_name=task_name, image_set=split, year=year, test=test_flag)    

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
