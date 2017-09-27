# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import json
import cv2 as cv
from xml.dom import minidom
from voc_eval import voc_eval
from fast_rcnn.config import cfg
from cfgs.arg_parser import config

class AllCats(imdb):
    def __init__(self, task_name, image_set, year, test=False, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        #os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)

        if not test:
            options = config(section_name=task_name+'_train', conf_file='lib/cfgs/detection.cfg')
        else:
            options = config(section_name=task_name+'_test', conf_file='lib/cfgs/detection.cfg') 

        self._classes = eval(options['classes'])

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes))) 
        self._image_ext = '.jpg'

        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        ########### Begin my own implementation
        # self._index_to_path = self.index_to_fullPath_dict('/home/liuhuawei/detection/annotate_data.txt')
        ## index to image absolute path
        self._index_to_path = {}
        ## index to image id
        self._index_to_id = {}
        ## image id to index 
        self._id_to_index = {}
        ## absolute path to json file
        self._json_file = options['input_json_file']
        if test:
            self._output_json_file = options['ouput_json_file']
        ## main function to convert json file into xml format
        self._json_to_xml(self._json_file)
        ## As we generate train/test txt file when converting json to xml
        self._image_index = self._load_image_set_index()
        ########### End of my own implementation
        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
         

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])


    ###### Using new function to do it
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = self._index_to_path[index] 
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path  

        
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    ## proposal method for 'RPN' network    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        ## return list of dicts    
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]

        ### here, we will not write to local file            
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    ## proposal method for 'selective_search method'
    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _json_to_xml(self, json_file):
        """Convert json file to xml file"""
        assert os.path.exists(json_file), \
                'json file not found at: {}'.format(json_file)
        with open(json_file, 'rb') as f:
            data = json.load(f)

        ## load image infos 
        ## 1. write index(without '.jpg') to txt file
        images = data['images'] 

        ## combine images
        # all_image_index = [img_dict['img'].split('.')[0] for img_dict in images]  
        all_image_index = [img_dict['img'] for img_dict in images]
        distinct_image_index = set(all_image_index)
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        print 'Writing txt file to %s' % image_set_file
        print 'There are %d different images' % len(distinct_image_index)
        with open(image_set_file, 'w') as f:
            for index in distinct_image_index:
                f.write(index+'\n')                                      

        ## 2. make index to abspath dict for indexing       
        # img_idx_abspath = [(img_dict['img'].split('.')[0], img_dict['file_name']) for img_dict in images]
        img_idx_abspath = [(img_dict['img'], img_dict['file_name']) for img_dict in images]
        self._index_to_path = dict(img_idx_abspath)


        ## load ground truth box infos
        annotations = data['annotations']

        combined_annts = {}
        ###
        for annot in annotations:
            # image_index = annot['source_id'].split('.')[0]
            image_index = annot['source_id']
            if image_index not in combined_annts.keys():
                combined_annts[image_index] = []
            combined_annts[image_index].append(annot)    

        assert len(combined_annts.keys()) == len(distinct_image_index)   

        ################2 changes
        # distinct_image_index = list(distinct_image_index)
        for idx, img_index in enumerate(distinct_image_index):
            # if idx < 8200:
            #     continue
            print "Converting json to xml %d/%d" % (idx+1, len(distinct_image_index))
            annots = combined_annts[img_index]
            img_ids = [an['image_id'] for an in annots]
            assert len(set(img_ids)) == 1

            self._id_to_index[img_ids[0]] = img_index
            self._index_to_id[img_index] = img_ids[0]            

            cat_ids = [an['category_id'] for an in annots]
            bboxes = np.array([an['bbox'] for an in annots])

            assert bboxes.shape[0] == len(img_ids)

            abspath = self._index_to_path[img_index]

            class_name = [self._classes[cat_id] for cat_id in cat_ids]
            save_folder = os.path.join(self._data_path, 'Annotations')
            save_name = img_index

            self._make_xml(abspath, img_index, class_name, bboxes, save_folder, save_name)     

    def _make_xml(self, img_full_path, img_relative_path, class_name, bbox, save_folder, save_name):      
        '''
        Make xml file for each image with specified infos.
        :param img_full_path: absolute path of the image
        :param img_relative_path: image index/name without '.jpg' suffix
        :class_name class for each bbox, we only support for one image now
        :param bbox: numpy array object, each row represents a bounding box
        :save_folder: where to save the xml file
        :save_name: final absolute path is "save_folder/save_name.xml"
        :return: None

        Note: This is only for one class detection, such as face/bag detection,
            for multi-class detection, we should modify this function.
        '''
        img_data = cv.imread(img_full_path)
        H, W, C = img_data.shape
        
        doc = minidom.Document()
        annot = doc.createElement("annotation")
        doc.appendChild(annot)
        
        folder = doc.createElement("folder")
        folder.appendChild(doc.createTextNode("BAG"))
        annot.appendChild(folder)
        
        filename = doc.createElement("filename")
        filename.appendChild(doc.createTextNode(img_relative_path))
        annot.appendChild(filename)
        
        size = doc.createElement("size")
        width = doc.createElement("width")
        width.appendChild(doc.createTextNode(str(W)))
        height = doc.createElement("height")
        height.appendChild(doc.createTextNode(str(H)))  
        depth = doc.createElement("depth")
        depth.appendChild(doc.createTextNode(str(C)))        
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        annot.appendChild(size)

        N, _ = bbox.shape
        bag_id = 0
        for i in xrange(N):        
            bag_id = bag_id + 1
            xmin = str(np.maximum(1, bbox[i, 0]))
            ymin = str(np.maximum(1, bbox[i, 1]))
            xmax = str(np.minimum(W, np.float(xmin)+bbox[i, 2]))
            ymax = str(np.minimum(H, np.float(ymin)+bbox[i, 3]))
        
            obj = doc.createElement("object")        
            name = doc.createElement("name")
            name.appendChild(doc.createTextNode(str(class_name[i])))
            id = doc.createElement("id")
            id.appendChild(doc.createTextNode(str(bag_id)))
       
            bndbox = doc.createElement("bndbox")
            xmin_ = doc.createElement("xmin")
            xmin_.appendChild(doc.createTextNode(xmin))
            ymin_ = doc.createElement("ymin")
            ymin_.appendChild(doc.createTextNode(ymin))
            xmax_ = doc.createElement("xmax")
            xmax_.appendChild(doc.createTextNode(xmax))
            ymax_ = doc.createElement("ymax")
            ymax_.appendChild(doc.createTextNode(ymax))
            
            bndbox.appendChild(xmin_)
            bndbox.appendChild(ymin_)
            bndbox.appendChild(xmax_)
            bndbox.appendChild(ymax_)
            
            obj.appendChild(name)
            obj.appendChild(id)                                                         
            obj.appendChild(bndbox)
            annot.appendChild(obj)
        
        f = file('%s/%s.xml' % (save_folder,save_name,), 'w')
        doc.writexml(f, indent='\t', addindent='\t', newl='\n')
        f.close()  

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        
        ############data preprocessing
        size = tree.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        ############
        
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            ################
            x1 = np.maximum(x1, 0)
            y1 = np.maximum(y1, 0)
            x2 = np.minimum(x2, width-1)
            y2 = np.minimum(y2, height-1)
            ################
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}  

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit2007/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    ##### write changes here    
    def _write_voc_results_file(self, all_boxes):
        json_list = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            # VOCdevkit2007/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                        xmin = int(dets[k, 0] + 1)
                        ymin = int(dets[k, 1] + 1)
                        w = int(dets[k, 2] - dets[k, 0] + 1)
                        h = int(dets[k, 3] - dets[k, 1] + 1)
                        json_list.append({'image_id':self._index_to_id[index], \
                            'category_id':self._class_to_ind[cls], \
                            'bbox':[xmin, ymin, w, h],\
                            'score':round(dets[k, -1], 3)})                 
        with open(self._output_json_file, 'wt') as f:
            f.write(json.dumps(json_list))                                           

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            ##voc_eval is "from voc_eval import voc_eval"
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            # print rec, prec, ap
            aps += [ap]
            print 'Overall recall for {} = {:.4f}'.format(cls, rec[-1])
            print 'Overall precision for {} = {:.4f}'.format(cls, prec[-1])
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    ### eg class = 5, num_images = 4, results will be like this, and 
    ### each element will be a N*5 array
    # [[[], [], [], []],
    #  [[], [], [], []],
    #  [[], [], [], []],
    #  [[], [], [], []],
    #  [[], [], [], []]] 

    ## output director: /output/default/test/  
        
        ## write all_boxes results to file 
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_bag.txt      
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            # print "i am on!!"
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            # print "i am off!!"
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.hand_bag import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
