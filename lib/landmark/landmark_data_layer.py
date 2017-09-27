import numpy as np
import cv2 as cv
import imgaug as ia
from imgaug import augmenters as iaa

import caffe 


class LandmarkDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str_)
        self._phase = str(self.phase)
        print self._phase
        self.crop_wh = int(params.get('crop_wh', 224))
         
        if self._phase == 'TRAIN':
            self.bbox_scale = float(params.get('bbox_scale', 1.0)) 
            self.use_augmentation = bool(params.get('use_augmentation', False)) 
        else:   
            self.bbox_scale = 1.0  
            self.use_augmentation = False

        self.base_dir = '/data/home/liuhuawei/tools/clothing_data/category-and-attribute-prediction/Img/'
        self.mean = np.array(params['mean'], dtype=np.float32).reshape((1, 1, 1, -1))

        self.train_data_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/train_data.txt'
        self.train_label_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/train_label.txt'
        self.train_landmark_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/train_landmark.txt'
        self.test_data_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_data.txt'
        self.test_label_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_label.txt'
        self.test_landmark_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_landmark.txt'            

        # self.train_data_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/t_data.txt'
        # self.train_label_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/t_label.txt'
        # self.train_landmark_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/t_landmark.txt'

        # self.test_data_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/t_data.txt'
        # self.test_label_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/t_label.txt'
        # self.test_landmark_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/t_landmark.txt'

        # self.train_data_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_data.txt'
        # self.train_label_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_label.txt'
        # self.train_landmark_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_landmark.txt'
        # self.test_data_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_data.txt'
        # self.test_label_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_label.txt'
        # self.test_landmark_file = '/data/home/liuhuawei/clothing_recognition/data/with_landmarks/val_landmark.txt'        

    
        print 'steup 1 !!!!!!!!!!!!!!!!!'
        if self._phase == 'TRAIN':
            self.train_batch_size = params['batch_size']
            self.train_img_paths, self.train_bbox, self.train_cats_id, self.train_attrs_id, self.train_landmark = \
                    self.load_data(self.train_data_file, self.train_label_file, self.train_landmark_file)
        else:     
            self.train_batch_size = params['batch_size']
            self.train_img_paths, self.train_bbox, self.train_cats_id, self.train_attrs_id, self.train_landmark = \
                    self.load_data(self.test_data_file, self.test_label_file, self.test_landmark_file)               

        print 'steup 2 !!!!!!!!!!!!!!!!!'                    
        # four tops: data, cats-label, attrs-label, landmarks 
        if len(top) != 4:
            raise Exception("Need to define four tops: data, cats-label, attrs-label, landmarks ")

        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.shuffle_inds()
        print 'steup 3 !!!!!!!!!!!!!!!!!' 
        # self.train_idx = 0
        # self.test_idx = 0
        top[0].reshape(16, 3, 224, 224)
        top[1].reshape(16)
        top[2].reshape(16, 1000)
        top[3].reshape(16, 24, 1, 1)

    def forward(self, bottom, top):
        blobs = self.get_next_minibatch()
        # ####################
        # new_blobs = blobs.copy()
        # for i in xrange(new_blobs['data'].shape[0]):
        #    img1 = (new_blobs['data'][i].transpose((1, 2, 0))[np.newaxis, :] + self.mean)[0].astype(np.uint8)
        #    # img1 = (new_blobs['data'][i].transpose((1, 2, 0))[np.newaxis, :])[0].astype(np.uint8)           
        #    img1[img1>255]=255
        #    img1[img1<0] = 0
        #    # print img1.shape
        #    landmark1 = new_blobs['landmark'][i].ravel()
        #    # print landmark1.shape
        #    visualize(img1, img1, landmark1, landmark1)
        # ######################

        top[0].reshape(*(blobs['data'].shape))
        top[0].data[...] = blobs['data'].astype(np.float32, copy=False)

        top[1].reshape(*(blobs['cat_label'].shape))
        top[1].data[...] = blobs['cat_label'].astype(np.int, copy=False) 

        top[2].reshape(*(blobs['attr_label'].shape))
        top[2].data[...] = blobs['attr_label'].astype(np.int, copy=False) 

        top[3].reshape(*(blobs['landmark'].shape))
        top[3].data[...] = blobs['landmark'].astype(np.int, copy=False)                        

        # for blob_name, blob in blobs.iteritems():
        #     top_ind = self._name_to_top_map[blob_name]
        #     # Reshape net's input blobs
        #     top[top_ind].reshape(*(blob.shape))
        #     # Copy data into net's input blobs
        #     top[top_ind].data[...] = blob.astype(np.float32, copy=False)        

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass    


    def load_data(self, data_file, attribute_file, landmark_file):
        img_paths = []
        bbox = []
        cats_id = []
        attrs_id = []
        landmark = []
        with open(data_file, 'r') as f_data, open(attribute_file, 'r') as f_attr, \
                open(landmark_file, 'r') as f_lan:
            data = f_data.readlines()
            for d in data:
                flags = d.strip('\n').split(' ')
                assert len(flags)==6
                img_paths.append(self.base_dir+flags[0])
                ## xmin, ymin, xmax, ymax format
                bbox.append([int(flags[1]), int(flags[2]), int(flags[3]), int(flags[4])])
                ## transform 1-based to 0-based category id
                cats_id.append(int(flags[5])-1)

            attrs_data = f_attr.readlines()
            for a in attrs_data:
                flags = a.strip('\n').split(' ')
                assert len(flags)==1000
                attrs_id.append(map(int, flags))

            landmark_data = f_lan.readlines()
            for l in landmark_data:
                flags = l.strip('\n').split(' ')[1:]
                assert len(flags)==24
                landmark.append(flags)
        assert len(img_paths)==len(bbox)==len(cats_id)==len(attrs_id)==len(landmark) 
        return img_paths, bbox, cats_id, attrs_id, landmark       

    def shuffle_inds(self):
        self.perm = np.random.permutation(np.arange(len(self.train_img_paths)))   
        self.train_idx = 0 

    def get_next_minibatch(self):
        cur_inds = self.get_next_minibatch_inds()
        minibatch_db = [self.perm[i] for i in cur_inds]
        return self.get_minibatch_blobs(minibatch_db)

    def get_next_minibatch_inds(self):
        if self.train_idx + self.train_batch_size > len(self.train_img_paths):
            self.shuffle_inds()
        cur_batch = self.perm[self.train_idx:self.train_idx+self.train_batch_size] 
        self.train_idx += self.train_batch_size    
        return  cur_batch        

    def get_minibatch_blobs(self, cur_inds):
        blobs = {'data':None, 'cat_label':None, 'attr_label':None, 'landmark':None} 
        blobs['data'] = [self.train_img_paths[ind] for ind in cur_inds]
        # for ind in cur_inds:
        #     print self.train_img_paths[ind] 
        blobs['cat_label'] = np.array([self.train_cats_id[ind] for ind in cur_inds], dtype=int)
        blobs['attr_label'] = np.array([self.train_attrs_id[ind] for ind in cur_inds], dtype=int)
        blobs['landmark'] = np.array([self.train_landmark[ind] for ind in cur_inds], dtype=int)
        ## xmin, ymin, xmax, ymax format
        bbox = np.array([self.train_bbox[ind] for ind in cur_inds], dtype=int)

        return self.transform_blobs(blobs, bbox)


    def transform_blobs(self, blobs, bbox):
        '''
        img_name: list of img_pathes
        bbox: list of bbox, each in [xmin, ymin, w, h] format
        resize_h,resize_w: resize the cropped image to specific shape
        mean_img: shape of (1, 1, 1, 3), channel means
        '''
        new_blobs = {}
        img_name = blobs['data']
        landmarks = blobs['landmark']
        assert len(img_name)==len(bbox)==len(landmarks)

        img_data, landmark = self.prepare_images_with_bbox(img_name, bbox, landmarks)  
        new_blobs['data'] = img_data
        new_blobs['cat_label'] = blobs['cat_label']
        new_blobs['attr_label'] = blobs['attr_label']
        new_blobs['landmark'] = landmark

        if self.use_augmentation:
            img_data = img_data.transpose((0, 2, 3, 1))
            # new_img_data = np.zeros(img_data.shap)
            for i in xrange(landmark.shape[0]):
                seq = augument_images()
                keypoints = []
                keypoints_on_images = []
                for j in range(0, len(landmark[i]), 3):
                    keypoints.append(ia.Keypoint(x=landmark[i][j+1], y=landmark[i][j+2]))
                keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=img_data[i].shape))
                # print img_data[i]   
                # cv.imshow('t1', img_data[i].astype(np.uint8))     
                # cv.waitKey(0)
                img_data[i] = seq.augment_images(img_data[i][np.newaxis, :])[0]
                # print kk.shape
                # kk = seq.augment_images(img_data[1:,:,::-1])
                # print img_data[0].shape
                # cv.imshow('t1', img_data[i].astype(np.uint8))     
                # cv.waitKey(0)
                    # print img_data[i]
                    # raise                
                    
                keypoints_aug = seq.augment_keypoints(keypoints_on_images)
                # print type(keypoints_aug[0].keypoints)
                # print type(keypoints_aug[0].keypoints[1])
                # raise
                for j in range(0, len(landmark[i]), 3):
                    landmark[i][j+1] = keypoints_aug[0].keypoints[j/3].x
                    landmark[i][j+2] = keypoints_aug[0].keypoints[j/3].y
            # print img_data.shape        
            new_blobs['data'] = img_data.transpose((0, 3, 1, 2)) 
            new_blobs['landmark'] = adjust_landmark(new_blobs['data'],landmark)
        return new_blobs           

    def prepare_images_with_bbox(self, img_name, bbox, landmarks):
        imgs = [cv.imread(name).astype(np.float32) for name in img_name]
        img_shape = np.array([im.shape for im in imgs], dtype=int)
        scaled_bbox = scale_bbox(bbox, self.bbox_scale)
        cliped_bbox = clip_boxes(scaled_bbox, img_shape) 

        for i in xrange(len(imgs)):
            img = imgs[i]
            H, W, C = img.shape
            ## elements in bbox in xmin, ymin, xmax, ymax format
            xmin = int(cliped_bbox[i][0])
            ymin = int(cliped_bbox[i][1])
            xmax = int(cliped_bbox[i][2])
            ymax = int(cliped_bbox[i][3])
            w = xmax - xmin + 1
            h = ymax - ymin + 1       
            w_ratio = self.crop_wh*1.0 / w
            h_ratio = self.crop_wh*1.0 / h        
            # imgs[i] = img[bbox[i][1]-1:bbox[i][1]+bbox[i][3], bbox[i][0]-1:bbox[i][0]+bbox[i][2], :]   
            # ####
            # ori_image = img.copy().astype(np.uint8)
            # ori_landmark = landmarks[i][:] 
            # ####
            # print xmin, xmax, ymin, ymax
            imgs[i] = img[ymin:ymax, xmin:xmax, :]     
            if landmarks is not None:
                landmark = landmarks[i][:]
                for j in np.arange(0, len(landmark), 3):
                    vis = landmark[j]
                    x_cor = landmark[j+1]
                    y_cor = landmark[j+2]
                    # -1: default value for mysql table, for up body, only 6 landmarks, rest 2 are annoted as -1
                    # 0: visiable
                    # 1: invisible/occluded
                    # 2: truncated/cut-off
                    assert vis in [-1, 0, 1, 2]
                    if vis==1 or vis==2:
                        continue
                    elif vis==-1 or x_cor<xmin or x_cor>xmax or y_cor<ymin or y_cor>ymax:
                        vis = 2
                    else:
                        x_cor = int((x_cor-xmin)*w_ratio)
                        y_cor = int((y_cor-ymin)*h_ratio)
                    landmarks[i][j] = vis
                    landmarks[i][j+1] = x_cor
                    landmarks[i][j+2] = y_cor
            # ####
            # visualize(ori_image, cv.resize(imgs[i].astype(np.uint8), (self.crop_wh, self.crop_wh)), np.array(ori_landmark, dtype=int), np.array(landmarks[i], dtype=int))
            # ####        
        # for i in xrange(len(imgs)):
        #     print imgs[i].shape   
        img = np.array([cv.resize(imgs[i], (self.crop_wh, self.crop_wh)) for i in xrange(len(imgs))])
        img -= self.mean
        img = img.transpose((0, 3, 1, 2))
        if landmarks is None:
            return img  
        else:
            return img, np.array(landmarks, dtype=int).reshape(len(landmarks), -1, 1, 1)         

def adjust_landmark(img_data, landmarks):
    for i in range(landmarks.shape[0]):
        _, img_h, img_w = img_data[i].shape
        landmark = landmarks[i]
        for j in np.arange(0, len(landmark), 3):
            vis = landmark[j]
            x_cor = landmark[j+1]
            y_cor = landmark[j+2]        
            assert vis in [0, 1, 2]
            if vis==0 and (x_cor<0 or x_cor>img_w or y_cor<0 or y_cor>img_h):  
                vis=2
            landmarks[i, j] = vis   
    return landmarks              
  
                    


def augument_images():
    # augumenter_id = np.random.randint(2)
    # if augumenter_id==0:
    #     seq = iaa.Sequential([iaa.Fliplr(0.5)])
    #     # print 0
    # elif augumenter_id==1:    
    seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Crop(percent=(0, 0.1))])
    #seq = iaa.Sequential([iaa.Fliplr(0.5)])
        # print 1
    # elif augumenter_id==2:        
        # seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 0.8))])  
        # print 2
    # else:
    #     print 3
    #     # seq = iaa.Sequential([iaa.Fliplr(0.5)])           
    # # else:    
    #     seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-15, 15))])  
    return seq.to_deterministic()        

def scale_bbox(bbox, scale_size):
    """
    Argus:
        -bbox: (x1, y1, x2, y2), shape = (1, 4)
    """
    scaled_bbox = np.zeros(bbox.shape)
    widths = bbox[:, 2] - bbox[:, 0] + 1.0
    heights = bbox[:, 3] - bbox[:, 1] + 1.0
    ctr_x = bbox[:, 0] + 0.5 * widths
    ctr_y = bbox[:, 1] + 0.5 * heights

    scaled_widths = widths * scale_size
    scaled_heights = heights * scale_size

    scaled_bbox[:, 0] = ctr_x - 0.5 * scaled_widths
    scaled_bbox[:, 2] = ctr_x + 0.5 * scaled_widths
    scaled_bbox[:, 1] = ctr_y - 0.5 * scaled_heights
    scaled_bbox[:, 3] = ctr_y + 0.5 * scaled_heights

    return scaled_bbox

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # print boxes.shape
    # print im_shape.shape
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[:, 1][:, np.newaxis] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[:, 0][:, np.newaxis] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[:, 1][:, np.newaxis] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[:, 0][:, np.newaxis] - 1), 0)
    return boxes 

def visualize(img1, img2, landmark1, landmark2):
    # print landmark1.shape
    # print landmark2.shape
    for i in np.arange(0, landmark1.shape[0], 3):
        if landmark1[i]==-1 or landmark1[i]==1 or landmark1[i]==2:
            continue
        cv.circle(img1, (int(landmark1[i+1]), int(landmark1[i+2])), 5,(0, 0, 225), -1)
        cv.imshow('test win', img1)
    cv.waitKey(0)
    # # win = cv.namedWindow('test win', flags=0)

    # for j in np.arange(0, landmark2.shape[0], 3):
    #     if landmark1[i]==-1 or landmark2[j]==1 or landmark2[j]==2:
    #         continue
    #     cv.circle(img2, (int(landmark2[j+1]), int(landmark2[j+2])), 5,(255, 0, 0), -1)
    #     cv.imshow('test win', img2)
    # cv.waitKey(0)       

