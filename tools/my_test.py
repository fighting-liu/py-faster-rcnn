#coding=utf-8
import cPickle as pickle
import json
import numpy as np

import _init_paths
import caffe, os, sys, cv2
from google.protobuf import text_format
import yaml

from collections import OrderedDict


def combine_json(json_1, json_2):
    with open(json_1, 'rb') as f:
        data_1 = json.load(f)
    print data_1.keys()
    print "~~~~~~~~~~~~~~~~"
    print len(data_1['images'])
    print data_1['images'][0]
    print data_1['images'][0].keys()

    print "~~~~~~~~~~~~~~"
    print len(data_1['annotations'])
    print data_1['annotations'][0]
    print data_1['annotations'][0].keys()

    print "~~~~~~~~~~~~~~"
    print data_1['categories']
    print data_1['categories'][0].keys()

    print "-----------------------------"
    with open(json_2, 'rb') as f:
        data_2 = json.load(f)

    print len(data_2['images'])
    # print type(data_2['images'])
    print len(data_2['annotations'])
    # print type(data_2['annotations'])
    data = {}
    data['categories'] = data_1['categories']
    data['images'] = data_1['images'] + data_2['images'][:2000]
    # data['annotations'] = data_1['annotations'] + data_2['annotations']
    data['annotations'] = data_1['annotations']
    handbag_img = [i['img'] for i in data_2['images'][:2000]]
    for ann in data_2['annotations']:
        if ann['source_id'] in handbag_img:
            data['annotations'].append(ann)

    print len(data['images'])
    print len(data['annotations'])

    img_ids = [d['id'] for d in data['images']]           
    print 'number of images', len(set(img_ids))


    with open('/data/home/liuhuawei/detection/input/all_cats_11cls_train_0427_plus_2000handbags.json', 'w') as f:
        f.write(json.dumps(data))

if __name__ == '__main__':
    json_1 = '/data/home/liuhuawei/detection/input/all_cats_11cls_train_0427.json'
    json_2 = '/data/home/liuhuawei/detection/input/7000_ann_from8650.json'
    combine_json(json_1, json_2)    


# old_prototxt = '/data/home/liuhuawei/clothing_recognition/model/vgg16_with_joints_test/deploy_backup.prototxt'
##Here, we have new layer names, which need to be initialized. We want to copy old params to new params.
# new_prototxt = '/data/home/liuhuawei/clothing_recognition/model/vgg16_with_joints_test/deploy.prototxt'

# ##Old model
# caffemodel = '/data/home/liuhuawei/clothing_recognition/model/vgg16_with_joints_test/new_vgg.caffemodel'
# # old_net = caffe.Net(old_prototxt, caffemodel, caffe.TEST)
# net = caffe.Net(new_prototxt, caffemodel, caffe.TEST)
# print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
# print net.blobs['pool5'].data.shape
# print net.blobs['fc6'].data.shape
# print net.params['fc6'][0].data.shape
# print net.params['fc6'][1].data.shape

# ##copy old_params values to new_params  values
# new_params = ['fc6_landmark', 'fc7_landmark']
# old_params = ['fc6', 'fc7']

# old_fc_params = {pr: (old_net.params[pr][0].data, old_net.params[pr][1].data) for pr in old_params}
# new_fc_params = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params}


# for pr, pr_conv in zip(old_params, new_params):
#     ##Copy for both W and b
#     new_fc_params[pr_conv][0][...] = old_fc_params[pr][0]
#     new_fc_params[pr_conv][1][...] = old_fc_params[pr][1]

# ##Make sure we have copied correctly.
# assert(old_net.params[op][0].data == new_net.params[np][0].data \
#          for op, np in zip(old_params, new_params))  

# ##Save new net
# new_net.save('/data/home/liuhuawei/clothing_recognition/model/vgg16_with_joints_test/new_vgg.caffemodel')

# model = '/data/home/liuhuawei/handbag_from_taobao/output/resnet77_by_2_on_21_style/v1__iter_33000.caffemodel'
# net = caffe.NetSpec()
# print net
# parser_object = caffe.proto.caffe_pb2.SolverParameter()
# file = open(net_prototxt, "rb")
# file_content = file.read()
# parser_object.ParseFromString(file_content)
# print parser_object
# parser_object = caffe.proto.caffe_pb2.NetParameter(net_prototxt)
# print parser_object

# a = caffe.proto.caffe_pb2.NetParameter()
# # print parser_object
# f = open(net_prototxt, 'r')
# # print dict(str(f.read()))
# text_format.Merge(str(f.read()), a)
# # print type(a.layer)
# # print type(a.layer[1])
# # a.layer[0].name = 'data_clothing_handbag'
# # print a.layer[]
# # print a.layer[0].name 
# # print a.layer[0].type
# # print a.layer[0].bottom
# # print a.layer[0].top

# # print type(a.layer[3])
# # print len(a.layer)
# # # a.to_proto()
# # print type(a)
# for i in xrange(len(a.layer)):
#     if len(a.layer[i].bottom) != 0:
#         # print a.layer[i].bottom
#         for j in range(len(a.layer[i].bottom)):
#             a.layer[i].bottom[j] += '_liuhuawei'
#             # print j
#         # print a.layer[i].bottom   
#         # print type(a.layer[i].bottom)
#         # a.layer[i].bottom = [j+'_liuhuawei' for j in a.layer[i].bottom]

#     if len(a.layer[i].top) != 0:
#         for j in range(len(a.layer[i].top)):
#             a.layer[i].top[j] += '_liuhuawei'        # print a.layer[i].top

#         # temp = [j+'_liuhuawei' for j in a.layer[i].top]   
#         # a.layer[i].top = temp
#     a.layer[i].name += '_liuhuawei'

# with open('./kkk.prototxt', 'w') as f:
#     f.write(str(a))

# print type(a)
# net = caffe.Net(net_prototxt)
# print net

# net = caffe.Net(net_prototxt, model, caffe.TEST)
# print net.params.keys()
# print type(net.layer)

# net_prototxt = '/data/home/liuhuawei/tools/pynetbuilder/test.prototxt'
# model = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/data/imagenet_models/resnet_50_1by2.caffemodel'
# net = caffe.Net(net_prototxt, model, caffe.TEST)
# caffe.set_mode_gpu()
# caffe.set_device(1)
# print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))


# print net.params['conv2'][0].data.shape
# print net.params['conv2'][1].data.shape
# print net.params['conv2'][2].data.shape


# print net.blobs['norm1'].data.shape
# print net.blobs['conv2'].data.shape
# caffe.set_device(1)
# old_prototxt = '/home/liuhuawei/vgg_deploy.prototxt'
# # ##Here, we have new layer names, which need to be initialized. We want to copy old params to new params.
# # new_prototxt = '/home/liuhuawei/new_vgg.prototxt'

# # ##Old model
# caffemodel = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel'
# net = caffe.Net(old_prototxt, caffemodel, caffe.TEST)
# # new_net = caffe.Net(new_prototxt, caffemodel, caffe.TEST)
# print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
# # print net.blobs['pool5'].data.shape
# # print net.blobs['fc6'].data.shape
# # print net.params['fc6'][0].data.shape
# # print net.params['fc6'][1].data.shape

# ##copy old_params values to new_params  values
# new_params = ['fc6_1X', 'fc7_1X', 'fc6_1.5X', 'fc7_1.5X']
# old_params = ['fc6', 'fc7', 'fc6', 'fc7']

# old_fc_params = {pr: (old_net.params[pr][0].data, old_net.params[pr][1].data) for pr in old_params}
# new_fc_params = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params}


# for pr, pr_conv in zip(old_params, new_params):
#     ##Copy for both W and b
#     new_fc_params[pr_conv][0][...] = old_fc_params[pr][0] 
#     new_fc_params[pr_conv][1][...] = old_fc_params[pr][1]

# ##Make sure we have copied correctly.
# assert(old_net.params[op][0].data == new_net.params[np][0].data \
#          for op, np in zip(old_params, new_params))  
      
# ##Save new net
# new_net.save('/home/liuhuawei/new_vgg.caffemodel')

# param_str  = 
# print yaml.load(param_str)
# import yaml
# document = """'a': 1 \n'b': 2"""
# print yaml.load(document)






# path_to_pkl = '/data/home/liuhuawei/detection/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_test/vgg16_faster_rcnn_iter_30000/bag_pr.pkl'
# fr = open(path_to_pkl)    
# inf = pickle.load(fr) 
# print inf
# print inf.keys()
# print len(inf['rec'])     
# print np.mean(inf['prec'])
# fr.close()            


# j_path = '/data/home/liuhuawei/handbag_from_taobao/data/5/2/0/2/520298739165.json'
# j_path = '/data/home/zengcheng/taobao_data/1/0/3/1/10317276493.json'

# # j_path = '/data/home/liuhuawei/handbag_from_taobao/style_data/1/1/2/1/11217299.json'
# with open(j_path, 'rb') as f:
#     data = json.load(f)

# print data.keys()
# print "~~~~~~~~~~~category~~~~~"
# print len(data[u'category'])
# print data['category']

# print "~~~~~~~~~~~title~~~~~"
# print len(data[u'title'])
# print data['title']

# print "~~~~~~~~~~~shopLink~~~~~"
# print len(data[u'shopLink'])
# print data['shopLink']

# print "~~~~~~~~~~~detail_url~~~~~"
# print len(data[u'detail_url'])
# print data[u'detail_url']

# print "~~~~~~~~~~~img_urls~~~~~"
# print len(data[u'img_urls'])
# print data['img_urls']

# print "~~~~~~~~~~~pic_url~~~~~"
# print len(data[u'pic_url'])
# print data['pic_url']

# print "~~~~~~~~~~~attributes~~~~~"
# print data['attributes']
# print len(data['attributes'])
# print data['attributes'][u'品牌']
# print data['attributes'][u'款式']
# print data['attributes'].get(u'流行款式名称', 'none')

# print "~~~~~~~~~~~nid~~~~~"
# print len(data['nid'])
# print data['nid']

# print "~~~~~~~~~~~display_urls~~~~~"
# print len(data[u'display_urls'])
# print data['display_urls']


# ####################
# j_path = '/data/home/liuhuawei/tools/coco_data/coco_zip_archiv/annotations/image_info_test2014.json'
# with open(j_path, 'rb') as f:
#     data = json.load(f)

# print data.keys() 
# print "~~~~~~~~~~~~~~info"
# print data['info']
# # print data['images'][0]
# # print data['images'][0].keys()
# print "~~~~~~~~~~~~~~licenses"
# print data['licenses']

# print "~~~~~~~~~~~~~~categories"
# print data['categories'][0].keys()

# print "~~~~~~~~~~~~~~annotations"
# print "annotation keys: ", data['annotations'][0].keys() 
# print data['annotations'][0]

# print "~~~~~~~~~~~~~~images"
# print "image keys: ", data['images'][0].keys() 
# print data['images'][0]
#print data['type']

# j_path = '/data/home/liuhuawei/clothing_train_sample_30000.json'

# j_path = '/data/home/liuhuawei/detection/input/clothing_all_cats_test_0307.json'
# with open(j_path, 'rb') as f:
#     input_data = json.load(f)
# print 'input data:'
# print input_data['categories']    

# j_path = '/data/home/liuhuawei/detection/output/all_cats_test_frcnn_det.json'
# with open(j_path, 'rb') as f:
#     target_data = json.load(f)
   
# print "Target:"  
# print type(target_data) 
# print len(target_data)
# print target_data[0]

# target_img_ids = set([d['image_id'] for d in target_data])
# print len(target_img_ids)
# print len(data['annotations'])  
# print data.keys()
# print "~~~~~~~~~~~~~~~~"
# print len(data['images'])
# print data['images'][0]
# print data['images'][0].keys()

# print "~~~~~~~~~~~~~~"
# print len(data['annotations'])
# print data['annotations'][0]
# print data['annotations'][0].keys()

# print "~~~~~~~~~~~~~~"
# print data['categories']
# print data['categories'][0].keys()

# j_path_handbag_train = '/home/liuhuawei/1000_ann_from8650.json'
# j_path_clothing_train = '/data/home/liuhuawei/clothing_test_sample.json'
# j_path_handbag_and_clothing_train = '/data/home/liuhuawei/hanbag1000_and_clothing3000_test.json'

# with open(j_path_handbag_train, 'rb') as f_handbag, \
#         open(j_path_clothing_train, 'rb') as f_clothing:
#     data_handbag = json.load(f_handbag)
#     data_clothing = json.load(f_clothing)
# data_handbag_image = data_handbag['images']    
# data_clothing_image = data_clothing['images']
# images = data_handbag_image + data_clothing_image
# print len(images)
# assert len(images) == len(data_clothing_image+data_handbag_image)

# data_handbag_anns = data_handbag['annotations']
# data_clothing_anns = data_clothing['annotations']
# for clothing_ann in data_clothing_anns:
#     clothing_ann['category_id'] += 1
# annotations = data_handbag_anns + data_clothing_anns
# assert len(annotations) == len(data_handbag_anns) + len(data_clothing_anns)

# categories = [{u'supercategory': u'none', u'id': 3, u'name': u'downbody'}, {u'supercategory': u'none', u'id': 2, u'name': u'upbody'}, \
# {u'supercategory': u'none', u'id': 4, u'name': u'fullbody'}, {u'supercategory': u'none', u'id': 1, u'name': u'bag'}]

# data = {}
# data['images'] = images
# data['annotations'] = annotations
# data['categories'] = categories
# # output_json_file = '/home/liuhuawei/detection_results.json'
# with open(j_path_handbag_and_clothing_train, 'wt') as f:
#     f.write(json.dumps(data)) 


# # output_json_file = '/home/liuhuawei/detection_results.json'
# # with open(output_json_file, 'wt') as f:
# #     f.write(json.dumps(a)) 

# j_path = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/output/faster_rcnn_end2end/coco_2007_test/VGG16_clothing_all_categories_0307_frcnn_iter_80000/detections_test2007_results_127ec082-c93f-41f2-b91a-e2e5c7245034.json'

# with open(j_path, 'rb') as f:
#     data = json.load(f)
# print '~~~~~~~~~~~~~~~~~~~~~~~~~'
# print "Now data:"
# print type(data)
# print len(data)
# print data[0]
# img_ids = set([d['image_id'] for d in data])
# print len(img_ids)


# for d in img_ids:
#     for dic in data:
#         if dic['image_id'] == d:
#             item =  dic 
#     print 'ours:'        
#     print item
#     for dic in target_data:
#         if dic['image_id'] == d:
#             target_item =  dic
#     print "target:"        
#     print target_item
#     print "gound truth:"
#     for dic in input_data['annotations']:
#         if dic['image_id'] == d:
#             print dic
# print data.keys()

# print data[0]
# print data['images'][0]
# print data['images'][0].keys()

# print "~~~~~~~~~~~~~~"
# print data['annotations'][0]
# print data['annotations'][0].keys()

# print "~~~~~~~~~~~~~~"
# print data['categories']
# print data['categories'][0].keys()


# def combine_prototxt(prototxt_1, output_file, add_name_1='_brand'):
#     from google.protobuf import text_format

#     ### Dealing the first model
#     net_param_1 = caffe.proto.caffe_pb2.NetParameter()    
#     f = open(prototxt_1, 'r')
#     text_format.Merge(str(f.read()), net_param_1)

#     ## for each of the layer object, it has attributes like 
#     ##             'type', 'bottom', 'top', 'name' and etc.
#     for i in xrange(len(net_param_1.layer)):
#         ## we don't change name of data layer
#         if net_param_1.layer[i].type == 'Input':
#             continue
#         ## change layer name    
#         net_param_1.layer[i].name += add_name_1    
#         ## change bottom name for each layer
#         if len(net_param_1.layer[i].bottom) != 0:
#             for j in range(len(net_param_1.layer[i].bottom)):
#                 net_param_1.layer[i].bottom[j] += add_name_1

#         ## change top layer for each layer        
#         if len(net_param_1.layer[i].top) != 0:
#             for j in range(len(net_param_1.layer[i].top)):
#                 net_param_1.layer[i].top[j] += add_name_1  
#     f.close()    
#     with open(output_file, 'w') as f:
#         f.write(str(net_param_1))                        

# def combine_caffemodel(prototxt_1, caffemodel_1, add_name_1='_brand', \
#                     prototxt_2=None, caffemodel_2=None, add_name_2='_style'):
#     caffe.set_device(1)
#     net = caffe.Net('./deploy.prototxt', caffemodel_1, caffe.TEST)

#     ## load the first model
#     net_1 = caffe.Net(prototxt_1, caffemodel_1, caffe.TEST) 
#     for key in net_1.params.keys():
#         for i in range(len(net_1.params[key])):
#             ## reset layer name and corresponding data for the ouput net
#             net.params[key+add_name_1][i].data[...] = net_1.params[key][i].data

#     ## load the second model         
#     net_2 = caffe.Net(prototxt_2, caffemodel_2, caffe.TEST) 
#     for key in net_2.params.keys():
#         for i in range(len(net_2.params[key])):
#             ## reset layer name and corresponding data for the ouput net
#             net.params[key+add_name_2][i].data[...] = net_2.params[key][i].data          

#     ## save the model params        
#     net.save('./deploy.caffemodel')          




# ##Old model
# caffemodel = '/data/home/liuhuawei/detection/dev-py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel'
# old_net = caffe.Net(old_prototxt, caffemodel, caffe.TEST)
# new_net = caffe.Net(new_prototxt, caffemodel, caffe.TEST)
# # print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
# # print net.blobs['pool5'].data.shape
# # print net.blobs['fc6'].data.shape
# # print net.params['fc6'][0].data.shape
# # print net.params['fc6'][1].data.shape

# ##copy old_params values to new_params  values
# new_params = ['fc6_1X', 'fc7_1X', 'fc6_1.5X', 'fc7_1.5X']
# old_params = ['fc6', 'fc7', 'fc6', 'fc7']

# old_fc_params = {pr: (old_net.params[pr][0].data, old_net.params[pr][1].data) for pr in old_params}
# new_fc_params = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params}


# for pr, pr_conv in zip(old_params, new_params):
#     ##Copy for both W and b
#     new_fc_params[pr_conv][0][...] = old_fc_params[pr][0] 
#     new_fc_params[pr_conv][1][...] = old_fc_params[pr][1]

# ##Make sure we have copied correctly.
# assert(old_net.params[op][0].data == new_net.params[np][0].data \
#          for op, np in zip(old_params, new_params))  
      
# ##Save new net
# new_net.save('/home/liuhuawei/new_vgg.caffemodel')

# def combine(handbag_data='/data/home/liuhuawei/output/handbag_from_taobao/train_taobao_and_huebloom.txt', \
#     clothing_data='/data/home/liuhuawei/clothing_recognition/data/clothing_train.txt'):

#     train_file = '/data/home/liuhuawei/hangbag_and_clothing_train.txt'
#     val_file = '/data/home/liuhuawei/hangbag_and_clothing_val.txt'
#     with open(handbag_data, 'rb') as f_handbag:
#         data = f_handbag.readlines()
#         permute = np.random.permutation(len(data))
#         train_index = permute[:30000]
#         val_index = permute[-3000:]
#         assert len(train_index)==30000 and len(val_index)==3000
#         train_data = [data[i] for i in train_index]
#         val_data = [data[i] for i in val_index]

#         # with open(train_file, 'w') as f_train, open(val_file, 'w') as f_val:
#         f_train = open(train_file, 'w')
#         f_val = open(val_file, 'w') 
#         for item in train_data:
#             flags = item.strip('\n').split(',')
#             path = flags[1]
#             bbox_1 = flags[2]
#             bbox_2 = flags[3]
#             bbox_3 = flags[4]
#             bbox_4 = flags[5]
#             f_train.write('0 '+path+' '+bbox_1+' '+bbox_2+' '+bbox_3+' '+bbox_4+'\n')

#         for item in val_data:
#             flags = item.strip('\n').split(',')
#             path = flags[1]
#             bbox_1 = flags[2]
#             bbox_2 = flags[3]
#             bbox_3 = flags[4]
#             bbox_4 = flags[5]
#             f_val.write('0 '+path+' '+bbox_1+' '+bbox_2+' '+bbox_3+' '+bbox_4+'\n')

#     base_prefix = '/data/home/liuhuawei/clothing_data/Img/'
#     with open(clothing_data, 'rb') as f_clothing:
#         data = f_clothing.readlines()
#         permute = np.random.permutation(len(data))
#         train_index = permute[:30000]
#         val_index = permute[-3000:]
#         assert len(train_index)==30000 and len(val_index)==3000
#         train_data = [data[i] for i in train_index]
#         val_data = [data[i] for i in val_index]

#         # with open(train_file, 'w') as f_train, open(val_file, 'w') as f_val:
#         for item in train_data:
#             flags = item.strip('\n').split(' ')
#             path = flags[0]
#             bbox_1 = flags[2]
#             bbox_2 = flags[3]
#             bbox_3 = flags[4]
#             bbox_4 = flags[5]
#             f_train.write('1 '+base_prefix+path+' '+bbox_1+' '+bbox_2+' '+bbox_3+' '+bbox_4+'\n')
#         for item in val_data:
#             flags = item.strip('\n').split(' ')
#             path = flags[0]
#             bbox_1 = flags[2]
#             bbox_2 = flags[3]
#             bbox_3 = flags[4]
#             bbox_4 = flags[5]
#             f_val.write('1 '+base_prefix+path+' '+bbox_1+' '+bbox_2+' '+bbox_3+' '+bbox_4+'\n')



# if __name__ == '__main__':
    # json_file = '/data/home/liuhuawei/hanbag1000_and_clothing3000_test.json'
    # # json_file = '/data/home/liuhuawei/clothing_test_sample.json'
    # with open(json_file, 'rb') as f:
    #     data = json.load(f)  
    # d = [ann['image_id'] for ann in data['annotations']]
    # print d 
    # image_id_ann = set([ann['image_id'] for ann in data['annotations']])
    # print len(image_id_ann)


    # json_file = '/home/liuhuawei/detection_hangbag1000_and_clothing3000_vgg.json'
    # json_file = '/data/home/liuhuawei/detection_clothing_test_sample.json'
    # with open(json_file, 'rb') as f:
    #     data = json.load(f)  
    # images =  data['images']    
    # annotations = data['annotations']
    # for im in images:
    #     flag = False
    #     for ann in annotations:
    #         if ann['source_id'] == im['img']:
    #             img_id = ann['image_id']
    #             flag = True
    #     assert flag
    #     im['id'] = img_id  
    # with open(json_file, 'wt') as f:
    #     f.write(json.dumps(data))               


    # kk = [ann for ann in data['images']] 
    # print kk[0]   
    # print data[0]    
    # image_id_dec = set([ann['image_id'] for ann in data])  
    # print len(image_id_dec)  
    # print image_id_dec
    # d = [ann['image_id'] for ann in data['annotations']]
    # print d 
    # print len(data)
    # print data[:5]


    # for annot in data['annotations']:
    #     category_id = annot['category_id']
    #     if int(category_id) > 1:
    #         annot['image_id'] = 'clothing_'+str(annot['image_id'])
    #     else:
    #         annot['image_id'] = 'hangbag_'+str(annot['image_id'])    
    # with open(json_file, 'wt') as f:
    #     f.write(json.dumps(data))                 
    # combine()
    # net = caffe.net()
    # net = caffe.Net('./deploy.prototxt', './deploy.caffemodel',caffe.TEST)
    # print net.params['conv_1_brand'][0].data
    # print net.layer[0]
    # # net_prototxt_style = '/data/home/liuhuawei/handbag_from_taobao/model/resnet77_by_2_on_21_style/deploy.prototxt'  
    # # net_prototxt_brand = '/data/home/liuhuawei/handbag_from_taobao/model/resnet77_by_2_brand/deploy.prototxt'
    # net_prototxt = '/data/home/liuhuawei/handbag_from_taobao/model/resnet77_by_2_on_21_style/deploy.prototxt'
    # output_file = './style_deploy.prototxt'
    # add_name_1 = '_style'
    # combine_prototxt(net_prototxt, output_file, add_name_1)
    
    # net_prototxt = './deploy.prototxt'
    # caffemodel = '/data/home/liuhuawei/handbag_from_taobao/output/resnet77_by_2_brand/v1__iter_59000.caffemodel'
    # caffe.set_device(1)
    # net = caffe.Net(net_prototxt, caffemodel, caffe.TEST)

    # # print("params {}".format(net.params.keys()))

    # print type(net.params['conv_stage3_block2_branch2c_style'])
    # print len(net.params['conv_stage3_block2_branch2c_style'])
    # print type(net.params['conv_stage3_block2_branch2c_style'][0])
    # print type(net.params['conv_stage3_block2_branch2c_style'][0].data)
    # # print net.params['conv_stage2_block2_branch2b_style'][0].data.shape



    # prototxt_1 = '/data/home/liuhuawei/handbag_from_taobao/model/resnet77_by_2_brand/deploy.prototxt'
    # prototxt_2 = '/data/home/liuhuawei/handbag_from_taobao/model/resnet77_by_2_on_21_style/deploy.prototxt' 
    # caffemodel_1 = '/data/home/liuhuawei/handbag_from_taobao/output/resnet77_by_2_brand/v1__iter_59000.caffemodel'
    # caffemodel_2 = '/data/home/liuhuawei/handbag_from_taobao/output/resnet77_by_2_on_21_style/v1__iter_33000.caffemodel'
    # combine_caffemodel(prototxt_1, caffemodel_1, add_name_1='_brand', \
    #                 prototxt_2=prototxt_2, caffemodel_2=caffemodel_2, add_name_2='_style')