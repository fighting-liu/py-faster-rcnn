# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        ## generate anchors the same way as we do in anchor target layer
        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        ## Add scale option    
        self._scale = np.array(layer_params.get('scale_param', False), np.float32)  
        ## Add global context option
        self._global_context = np.array(layer_params.get('global_context', False), np.float32) 

        assert self._scale == 0.0 or self._global_context == 0.0, 'Only one of them can be True'                     

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        ## If scale_param is True, we need to add top output
        if self._scale != 0.0 or self._global_context != 0.0:
            top[1].reshape(1, 5)

        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        # 1.generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # 2.clip predicted boxes to image
        # 3.remove predicted boxes with either height or width < threshold
        # 4.sort all (proposal, score) pairs by score from highest to lowest
        # 5.take top pre_nms_topN proposals before NMS
        # 6.apply NMS with threshold 0.7 to remaining proposals
        # 7.take after_nms_topN proposals after NMS
        # 8.return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print "DEBUG for ProposalLayer"
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        ######In case, proposals and scores are empty
        if len(scores) == len(proposals) == 0:
            proposals = np.array([0, 0, 1, 1]).reshape((1, 4))
            scores = np.array([0], dtype=np.float32)
        # print proposals
        # print scores        
        # print proposals.shape

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob
        # print blob
        if self._scale != 0.0:
            ################### Add context infomations ##############$
            ## Here we hard code the scale size to be 1.5X
            scaled_rois = _change_roi_size(blob, self._scale)
            scaled_rois[:, 1:] = clip_boxes(scaled_rois[:, 1:], im_info[:2]) 

        if self._global_context != 0.0:
            ### we use the same interface here (scaled_rois)
            scaled_rois = np.zeros(blob.shape)        
            assert scaled_rois.shape[1] == 5 and np.all(scaled_rois[:, 0] == 0)
            # scaled rois (0, x1, y1, x2, y2)
            scaled_rois[:, 1] = 0
            scaled_rois[:, 2] = 0
            scaled_rois[:, 3] = im_info[1] - 1
            scaled_rois[:, 4] = im_info[0] - 1 
            
        if self._scale != 0.0 or self._global_context != 0.0:  
            top[1].reshape(*(scaled_rois.shape))
            top[1].data[...] = scaled_rois     
            # print scaled_rois.shape
            ################### End
        # # [Optional] output scores blob
        # if len(top) > 1:
        #     top[1].reshape(*(scores.shape))
        #     top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

###### fix: change rois with respect to different scale
def _change_roi_size(filtered_rois, scale_size):
    """Change roi size with respect to the given scale size

    Params:
        filtered_rois: 128 rois for training, eg [0, x1, y1, x2, y2]
        scale_size: to which scale to resize the rois.    
    """
    scaled_rois = np.zeros(filtered_rois.shape)
    
    widths = filtered_rois[:, 3] - filtered_rois[:, 1] + 1.0
    heights = filtered_rois[:, 4] - filtered_rois[:, 2] + 1.0
    ctr_x = filtered_rois[:, 1] + 0.5 * widths
    ctr_y = filtered_rois[:, 2] + 0.5 * heights

    scaled_widths = widths * scale_size
    scaled_heights = heights * scale_size

    scaled_rois[:, 1] = ctr_x - 0.5 * scaled_widths
    scaled_rois[:, 3] = ctr_x + 0.5 * scaled_widths
    scaled_rois[:, 2] = ctr_y - 0.5 * scaled_heights
    scaled_rois[:, 4] = ctr_y + 0.5 * scaled_heights

    return scaled_rois

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
