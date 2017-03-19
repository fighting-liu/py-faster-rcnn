# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

# from datasets.pascal_voc import pascal_voc
# from datasets.aflw_face import pascal_voc
# from datasets.hand_bag_old import pascal_voc
from datasets.clothing import Clothing
from datasets.hand_bag import Handbag
from datasets.handbag_and_clothing import Handbag_And_Clothing
from datasets.all import AllCats
from datasets.coco_all import coco
# from datasets.clothing import pascal_voc
# from datasets.coco import coco
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
# for year in ['2007', '2012']:
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'clothing_{}_{}'.format(year, split)
#         if split == 'test':
#             __sets[name] = (lambda split=split, year=year: Clothing(split, year, test=True))
#         else:
#             __sets[name] = (lambda split=split, year=year: Clothing(split, year))


# for year in ['2007', '2012']:
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'handbag_{}_{}'.format(year, split)
#         if split == 'test':
#             __sets[name] = (lambda split=split, year=year: Handbag(split, year, test=True))
#         else:
#             __sets[name] = (lambda split=split, year=year: Handbag(split, year)) 

# for year in ['2007', '2012']:
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'handbag_and_clothing_{}_{}'.format(year, split)
#         if split == 'test':
#             __sets[name] = (lambda split=split, year=year: Handbag_And_Clothing(split, year, test=True))
#         else:
#             __sets[name] = (lambda split=split, year=year: Handbag_And_Clothing(split, year)) 

# for year in ['2007', '2012']:
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'all_categories_{}_{}'.format(year, split)
#         if split == 'test':
#             __sets[name] = (lambda split=split, year=year: AllCats(split, year, test=True))
#         else:
#             __sets[name] = (lambda split=split, year=year: AllCats(split, year))                         


     

# # Set up coco_2014_<split>
# for year in ['2014']:
#     for split in ['train', 'val', 'minival', 'valminusminival']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))

# # Set up coco_2015_<split>
# for year in ['2015']:
#     for split in ['test', 'test-dev']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))

# def get_imdb(name):
#     """Get an imdb (image database) by name."""
#     if not __sets.has_key(name):
#         raise KeyError('Unknown dataset: {}'.format(name))
#     return __sets[name]()
def get_imdb(name):
    """Get an imdb (image database) by name.
       previous name format: 2007_trainval(test)_taskname
       now name format: voc(coco)_2007_trainval(test)_taskname
    """
    # year, split, task_name = name.split('_', 2)
    # print task_name, split, year
    # assert split in ['trainval', 'test'] and year == '2007'
    # test_flag = True if split == 'test' else False
    # return AllCats(task_name=task_name, image_set=split, year=year, test=test_flag)
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
