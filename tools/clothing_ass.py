def load_bbox(bbox_file):
    img_to_bbox = {}
    with open(bbox_file, 'rb') as f:
        lines = f.readlines()
        for line in lines[2:]:
            flag = line.strip('\n').split(' ')
            img_to_bbox[flag[0]] = [int(flag[-4]), int(flag[-3]), int(flag[-2]), int(flag[-1])]
    return img_to_bbox        
   

def get_trainset(data_path):
    train_val = []
    with open(data_path, 'rb') as f:
        lines = f.readlines()
        for line in lines[2:]:
            flag = line.strip('\n').split(' ')
            if flag[-1] == 'train':
                train_val.append(flag[0])
    print "num of trainval is %d" % len(train_val)            
    return train_val

def load_category(data_path, top_thresh, down_thresh, full_thresh):
    img_to_category = {}
    with open(data_path, 'rb') as f:
        lines = f.readlines()
        for line in lines[2:]:
            flag = line.strip('\n').split(' ')
            if int(flag[-1]) <= top_thresh:c
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

if __name__ == '__main__':
    # file_path = '/data/home/liuhuawei/clothing_data/Eval/list_eval_partition.txt'
    # bbox_path = '/data/home/liuhuawei/clothing_data/Anno/list_bbox.txt'
    # category_path = '/data/home/liuhuawei/clothing_data/Anno/list_category_img.txt'


    # train_val = get_trainset(file_path)
    # ##1-20
    # top_thresh = 20
    # ##21-36
    # down_thresh = 36
    # ##37-50
    # full_thresh = 50

    # img_to_bbox = load_bbox(bbox_path)
    # img_to_category = load_category(category_path, top_thresh, down_thresh, full_thresh)
             
    # wtf_path = '/data/home/liuhuawei/clothing.txt'
    # write_new_file(train_val, img_to_bbox, img_to_category, wtf_path)                       




