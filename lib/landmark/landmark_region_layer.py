import caffe
import numpy as np
import yaml
import sys

class LandmarkRegionLayer(caffe.Layer):
    """docstring for ClassName"""
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._side_length_w = layer_params.get('side_length_w', 28)
        self._side_length_h = layer_params.get('side_length_h', 28)

        batch_size = bottom[0].data.shape[0]
        # (index, x1, y1, x2, y2)
        top[0].reshape(batch_size*8, 5)


    def forward(self, bottom, top):
        # all_landmarks = bottom[0]
        # all_landmarks = np.random.random((8, 14, 2)) 
        all_landmarks = bottom[0].data 
        # print all_landmarks.shape
        # raise
        # print '#########Land mark:'
        # print all_landmarks[0, :2, 0, 0]
        # print all_landmarks[]
        side_length_w = self._side_length_w 
        side_length_h = self._side_length_h
        img_width = 224
        img_height = 224
        batch_size, num_cors, _, _ = all_landmarks.shape 
        num_parts = num_cors / 3 #vis,x,y
        ## batch_index, x1, y1, x2, y2 
        all_rois = np.zeros((batch_size*num_parts, 5))
        for  i in xrange(batch_size):
            for j in xrange(num_parts):
                vis = all_landmarks[i, j*3]
                x_ctr = all_landmarks[i, j*3+1, 0, 0]
                y_ctr = all_landmarks[i, j*3+2, 0, 0]
                all_rois[i*num_parts+j, 0] = i
                # if x_ctr == 0 or y_ctr == 0:
                if vis==1 or vis==2:
                    all_rois[i*num_parts+j, 1:] = np.array([-500, -500, -500, -500])
                    # all_rois[i*num_parts+j, 1:] = np.array([-1, -1, -1, -1])
                else:    
                    all_rois[i*num_parts+j, 1:] = _make_region(x_ctr, y_ctr, side_length_w, side_length_h)
        # all_rois[:, 1:] = _clip_boxes(all_rois[:, 1:], img_width, img_height)
	#print all_landmarks
	#print all_rois
	#raise
        top[0].reshape(*all_rois.shape)
        top[0].data[...] = all_rois  
        # print all_rois[0]
        # print all_rois
        # print all_rois.shape    

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
