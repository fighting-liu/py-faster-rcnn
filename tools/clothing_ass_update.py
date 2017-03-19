import numpy as np
import cv2 as cv
import json
from collections import defaultdict
import os

def load_bbox(bbox_file):
    img_to_bbox = {}
    with open(bbox_file, 'rb') as f:
        lines = f.readlines()
        for line in lines[2:]:
            flag = line.strip('\n').split(' ')
            img_to_bbox[flag[0]] = [int(flag[-4]), int(flag[-3]), int(flag[-2]), int(flag[-1])]
    return img_to_bbox        
   
def get_trainset(data_path, name):
    train_val = []
    with open(data_path, 'rb') as f:
        lines = f.readlines()
        for line in lines[2:]:
            flag = line.strip('\n').split(' ')
            if flag[-1] == name:
                train_val.append(flag[0])
    print "num of %s is %d" % (name, len(train_val))            
    return train_val

def load_category(data_path, top_thresh, down_thresh, full_thresh):
    img_to_category = {}
    with open(data_path, 'rb') as f:
        lines = f.readlines()
        for line in lines[2:]:
            flag = line.strip('\n').split(' ')
            if int(flag[-1]) <= top_thresh:
                img_to_category[flag[0]] = 1
            elif int(flag[-1]) <= down_thresh:
                img_to_category[flag[0]] = 2
            else:  
                img_to_category[flag[0]] = 3  
    return img_to_category            

def write_new_file(train_val, img_to_bbox, img_to_category, wtf_path):
    with open(wtf_path, 'w') as f:
        for idx, img in enumerate(train_val):
            print "Processing %d/%d!!!!" % (idx+1, len(train_val))
            category_id = img_to_category[img]
            bbox = img_to_bbox[img]
            f.write(img+' '+str(category_id)+' '+str(bbox[0])+\
                ' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n')

def sample_data(input_file, sample_num, output_file):
    avg_num = sample_num / 3
    cate_one = avg_num
    cate_two = avg_num
    cate_three = sample_num - avg_num*2

    idx_one = []
    idx_two = []
    idx_three = []
    with open(input_file, 'rb') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            id_num = int(line.split(' ')[1])
            if id_num == 1:
                idx_one.append(idx)
            elif id_num == 2:
                idx_two.append(idx)
            elif id_num == 3:
                idx_three.append(idx)
            else:
                print 'error'
    idx_one = np.array(idx_one)[np.random.permutation(len(idx_one))[:cate_one]] 
    idx_two = np.array(idx_two)[np.random.permutation(len(idx_two))[:cate_two]]
    idx_three = np.array(idx_three)[np.random.permutation(len(idx_three))[:cate_three]]

    with open(input_file, 'rb') as f_read, open(output_file, 'w') as f_write:
        lines = f_read.readlines()
        for idx, line in enumerate(lines):
            if (idx in idx_one) or (idx in idx_two) or (idx in idx_three):
                f_write.write(line)

def sta_aspect_ratio():
    stas = defaultdict(list)
    with open('/data/home/liuhuawei/clothing_train.txt', 'rb') as f:        
        lines = f.readlines()
        for line in lines:
            flag = line.split(' ')
            cat_id = int(flag[1])
            xmin = int(flag[2])
            ymin = int(flag[3])
            w = float(flag[4]) - xmin + 1
            h = float(flag[5]) - ymin + 1
            w_to_h = w / h
            if cat_id == 2 and w_to_h >= 1:
                stas[4].append(w_to_h)
            # if cat_id == 3 and w_to_h >= 1.75:
            #     stas[5].append(w_to_h)    
            else:        
                stas[cat_id].append(w_to_h)
    for key in stas.keys():
        print key, len(stas[key]), sum(stas[key]) / len(stas[key])

def sta_gt_bbox_area():
    aspect_ratio_stas = defaultdict(list)     
    bbox_area_stas = defaultdict(list)
    infix = '/data/home/liuhuawei/clothing_data/Img/'
    target_size = 600
    max_size = 1000

    with open('/data/home/liuhuawei/clothing_train.txt', 'rb') as f:        
        lines = f.readlines()
        for idx, line in enumerate(lines):
            print "Processing %d/%d!!!" % (idx+1, len(lines))

            flag = line.split(' ')
            file_name = flag[0]
            cat_id = int(flag[1])
            xmin = int(flag[2])
            ymin = int(flag[3])
            w = float(flag[4]) - xmin + 1
            h = float(flag[5]) - ymin + 1 

            img = cv.imread(os.path.join(infix, file_name))
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            target_w = im_scale * w
            target_h = im_scale * h 

            w_to_h = w / h
            if cat_id == 2 and w_to_h >= 1:
                aspect_ratio_stas[4].append(w_to_h) 
                bbox_area_stas[4].append((target_w, target_h))  
            else:        
                aspect_ratio_stas[cat_id].append(w_to_h)            
                bbox_area_stas[cat_id].append((target_w, target_h)) 
    for key in aspect_ratio_stas.keys():
        print key, len(aspect_ratio_stas[key]), \
            sum(aspect_ratio_stas[key]) / len(aspect_ratio_stas[key]) 
    
    for key in bbox_area_stas.keys():
        value = np.array(bbox_area_stas[key])
        print key, value.shape[0], value.mean(axis=0)

def sta_cat_data():
    with open('/data/home/liuhuawei/clothing_data/Anno/list_category_img.txt', 'rb') as f:
        cat_res = np.zeros(50, dtype=np.int32)

        lines = f.readlines()
        for line in lines[2:]:
            ## 0-index based 
            id_num = int(line.split(' ')[-1]) - 1
            cat_res[id_num] += 1
        print cat_res
        print np.where(cat_res>500)[0]
        print np.where(cat_res>1000)[0]
        print np.sum(cat_res)
        np.savetxt('/data/home/liuhuawei/clothing_data/Anno/cat_stas.txt', cat_res, fmt='%d', delimiter=',')

def sta_attr_data():
    with open('/data/home/liuhuawei/clothing_data/Anno/list_attr_img.txt', 'rb') as f:
        attr_res = np.zeros(1000, dtype=np.int32)

        lines = f.readlines()
        # flag = lines[2].strip().split(' ')
        # print len(flag)
        # print flag
        # print flag[-1]
        # print flag[-1000]
        # print flag[720:750]
        # # print len(lines)
        # # print lines[2]
        for idx, line in enumerate(lines[2:]):
            print "Processing %d/%d!!!" % (idx+1, len(lines))
            flag = line.strip('\n').split(' ')
            flag = [k for k in flag[1:] if k != '']
            assert len(flag) == 1000, len(flag)
            for i in xrange(1000):
                id_str = flag[i]
                id_num = int(id_str)
                assert id_num == 1 or id_num == -1, id_num
                attr_res[i] += id_num == 1
        print np.where(attr_res>=1000)[0]
        print len(np.where(attr_res>=1000)[0])
        np.savetxt('/data/home/liuhuawei/clothing_data/Anno/tmp.txt', attr_res, fmt='%d', delimiter=',')        
        # print attr_res
        print np.sum(attr_res)        

def txt_to_json(txt_file, category_to_id, output_json_file):
    infix = '/data/home/liuhuawei/clothing_data/Img/'
    json_dict = {}
    images = []
    annotations = []
    categories = []
    for category in category_to_id:
        cat_dict = {}
        cat_dict['supercategory'] = 'none'
        cat_dict['id'] = category_to_id[category]
        cat_dict['name'] = category
        categories.append(cat_dict)
 
    with open(txt_file, 'rb') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            print 'preprocessing %d/%d for json!!!' % (idx+1, len(lines))
            flag = line.strip('\n').split(' ')
            img_name_with_jpg = flag[0]
            category_id = int(flag[1])
            xmin = int(flag[2])
            ymin = int(flag[3])
            w = int(flag[4]) - xmin + 1
            h = int(flag[5]) - ymin + 1
            bbox = [xmin, ymin, w, h]
            img_full_path = infix + img_name_with_jpg
            # img_relative_path = img_name_with_jpg.split('.')[0]
            img_relative_path = img_name_with_jpg.replace('/', '_')   
            img_data = cv.imread(img_full_path)
            H, W, _ = img_data.shape  

            img_dict = {}
            img_dict['img'] = img_relative_path
            img_dict['file_name'] = img_full_path
            img_dict['height'] = H 
            img_dict['width'] = W
            img_dict['url'] = 'none'
            img_dict['id'] = idx+1
            images.append(img_dict)

            anno_dict = {}
            anno_dict['segmentation'] = 'none'
            anno_dict['area'] = h * w
            anno_dict['iscrowd'] = 0
            anno_dict['image_id'] = idx + 1
            anno_dict['bbox'] = bbox
            anno_dict['source_id'] = img_relative_path
            anno_dict['category_id'] = category_id
            anno_dict['id'] = idx + 1
            annotations.append(anno_dict)
    json_dict['images'] = images 
    json_dict['annotations'] = annotations
    json_dict['categories'] = categories   

    with open(output_json_file, 'wt') as f:
        f.write(json.dumps(json_dict)) 

    

if __name__ == '__main__':
    # # sta_aspect_ratio()
    # # sta_gt_bbox_area()
    # sta_cat_data()
    # # sta_attr_data()

    # # sta_data()
    ## Path to the partition file
    file_path = '/data/home/liuhuawei/clothing_data/Eval/list_eval_partition.txt'
    ## Path to the bbox file
    bbox_path = '/data/home/liuhuawei/clothing_data/Anno/list_bbox.txt'
    ## Path to category-img file, each image labels a category in [1,2,3] for ['upbody', 'downbody', 'fullbody'] 
    category_path = '/data/home/liuhuawei/clothing_data/Anno/list_category_img.txt'

    ## get dataset of "name"
    name = 'train'
    data = get_trainset(file_path, name)
    ##1-20
    top_thresh = 20
    ##21-36
    down_thresh = 36
    ##37-50
    full_thresh = 50

    # ## get bbox and category dicts for images
    # img_to_bbox = load_bbox(bbox_path)
    # img_to_category = load_category(category_path, top_thresh, down_thresh, full_thresh)
             
    # ## write txt file with each row like 'img_name category_id xmin ymin xmax ymax'         
    wtf_path = '/data/home/liuhuawei/clothing_%s.txt' % name
    # write_new_file(data, img_to_bbox, img_to_category, wtf_path)  

    ## sample data in test set
    sample_num = 30000
    wtf_path_sample = '/data/home/liuhuawei/clothing_%s_sample_%d.txt' % (name, sample_num,)
    sample_data(wtf_path, sample_num, wtf_path_sample)

    ## convert txt file to json file
    output_json_file = '/data/home/liuhuawei/clothing_%s_sample_%d.json' % (name, sample_num,)
    category_to_id = dict([('upbody',1) ,('downbody',2), ('fullbody', 3)]) 
    txt_to_json(wtf_path_sample, category_to_id, output_json_file)    









