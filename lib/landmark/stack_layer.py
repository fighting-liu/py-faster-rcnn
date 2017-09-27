import caffe
import numpy as np
import yaml
import sys

class StackLayer(caffe.Layer):
    """docstring for ClassName"""
    def setup(self, bottom, top):
        # (index, x1, y1, x2, y2)
        top[0].reshape(1, (512*9), 7, 7)


    def forward(self, bottom, top):
        # all_landmarks = bottom[0]
        # all_landmarks = np.random.random((8, 14, 2)) 
        all_landmarks = bottom[0].data 
        print '#########Land mark shape :'
        print all_landmarks.shape
        print all_landmarks
        side_length_w = self._side_length_w 
        side_length_h = self._side_length_h
        img_width = 224
        img_height = 224
        batch_size, num_cors, _, _ = all_landmarks.shape 
        num_parts = num_cors / 2
        ## batch_index, x1, y1, x2, y2 
        all_rois = np.zeros((batch_size*num_parts, 5))
        for  i in xrange(batch_size):
            for j in xrange(num_parts):
                x_ctr = all_landmarks[i, 2*j, 0, 0]
                y_ctr = all_landmarks[i, 2*j+1, 0, 0]
                all_rois[i*num_parts+j, 0] = i
                if x_ctr == 0 or y_ctr == 0:
                    all_rois[i*num_parts+j, 1:] = np.array([-500, -500, 0, 0])
                else:    
                    all_rois[i*num_parts+j, 1:] = _make_region(x_ctr, y_ctr, side_length_w, side_length_h)
        # all_rois[:, 1:] = _clip_boxes(all_rois[:, 1:], img_width, img_height) 

        top[0].reshape(*all_rois.shape)
        top[0].data[...] = all_rois   
        print all_rois
        print all_rois.shape    

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


