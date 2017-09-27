import caffe
import numpy as np
import yaml
import sys

class TestLayer(caffe.Layer):
    """docstring for ClassName"""
    def setup(self, bottom, top):

        # (index, x1, y1, x2, y2)
        
        top[0].reshape(1)
        # raise

    def forward(self, bottom, top):
        # pass
        print 'rois data(all zeros?):'
        # print np.all(bottom[0].data[0] == 0)
        print bottom[0].data.shape
        # print bottom[0].data.shape
        # print bottom[0].data[0]
        # print bottom[0].data[0].shape
        # print np.all(bottom[0].data[0]==0) 
        # print np.where(bottom[0].data[0][0] == 0)[0][0]
        # rois_data = bottom[0].data
        # all_landmarks = bottom[1].data 
        # # print all_landmarks[]
        # img_width = 224
        # img_height = 224
        # batch_size, num_cors, _, _ = all_landmarks.shape 
        # num_parts = num_cors / 2
        # ## batch_index, x1, y1, x2, y2 
        # all_rois = np.zeros((batch_size*num_parts, 5))
        # for  i in xrange(batch_size):
        #     for j in xrange(num_parts):
        #         x_ctr = all_landmarks[i, 2*j, 0, 0]
        #         y_ctr = all_landmarks[i, 2*j+1, 0, 0]
        #         all_rois[i*num_parts+j, 0] = i
        #         if (x_ctr == 0 or y_ctr == 0) and j==0:
        #             # print rois_data[j]
        #             # print rois_data[0, 28]
        #             # print rois_data[0, 119]
        #             print '~~~~~~~~~~~~~~~~~~~'
        #             print np.all(rois_data[j] == 0)
        #             # print np.where(rois_data[j] != 0)
                    # raise

        b_all_rois = np.array(1)
        top[0].reshape(*b_all_rois.shape)
        top[0].data[...] = b_all_rois 

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _clip_boxes(boxes, img_width, img_height):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], img_width - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], img_height - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], img_width - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], img_height - 1), 0)
    return boxes

def _make_region(x_ctr, y_ctr, side_length_w, side_length_h):
    x1 = x_ctr - side_length_w/2
    x2 = x_ctr + side_length_w - side_length_w/2
    y1 = y_ctr - side_length_h/2
    y2 = y_ctr + side_length_h - (side_length_h/2)

    return np.array([x1, y1, x2, y2])