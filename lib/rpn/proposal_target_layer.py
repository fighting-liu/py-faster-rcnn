# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform, clip_boxes
from utils.cython_bbox import bbox_overlaps

DEBUG = True

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._iter = 0

        ## Add scale option 
        self._scale = np.array(layer_params.get('scale_param', False), np.float32)
        ## Add global context option
        self._global_context = np.array(layer_params.get('global_context', False), np.float32)

        assert self._scale == 0.0 or self._global_context == 0.0, 'Only one of them can be True'
        
        ## If scale_param is not False, we need to add top output
        if self._scale != 0.0 or self._global_context != 0.0:
            # sampled rois (0, x1, y1, x2, y2)
            top[0].reshape(1, 5)
            # scaled rois (0, x1, y1, x2, y2)
            top[1].reshape(1, 5)
            # labels
            top[2].reshape(1, 1)
            # bbox_targets
            top[3].reshape(1, self._num_classes * 4)
            # bbox_inside_weights
            top[4].reshape(1, self._num_classes * 4)
            # bbox_outside_weights
            top[5].reshape(1, self._num_classes * 4)
        else:
            # sampled rois (0, x1, y1, x2, y2)
            top[0].reshape(1, 5)
            # labels
            top[1].reshape(1, 1)
            # bbox_targets
            top[2].reshape(1, self._num_classes * 4)
            # bbox_inside_weights
            top[3].reshape(1, self._num_classes * 4)
            # bbox_outside_weights
            top[4].reshape(1, self._num_classes * 4)  

        if DEBUG:
            print "ProposalTargetLayer!"
            # number of images
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

            ## below are for bbox targets statisticals for all classes (except background)
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            # number of positive rois 
            self._counts = cfg.EPS    

            ##bbox targets statisticals for each class(except background)
            self._cls_sums = np.zeros((self._num_classes-1, 4))
            self._cls_squared_sums = np.zeros((self._num_classes-1, 4))
            self._cls_counts = np.ones(self._num_classes-1) * cfg.EPS            

    def forward(self, bottom, top):
        self._iter += 1
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        ################### Add image shape to clip the box ##########
        # im_info = bottom[2].data
        im_info = bottom[2].data[0, :]
        ################### End

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)
        
        if self._scale != 0.0:
            ################### Add context infomations ##############$
            scaled_rois = _change_roi_size(rois, self._scale)
            scaled_rois[:, 1:] = clip_boxes(scaled_rois[:, 1:], im_info) 
            ################### 

        if self._global_context != 0.0:
            ### we use the same interface here (scaled_rois)
            scaled_rois = np.zeros(rois.shape)        
            assert scaled_rois.shape[1] == 5 and np.all(scaled_rois[:, 0] == 0)
            # scaled rois (0, x1, y1, x2, y2)
            scaled_rois[:, 1] = 0
            scaled_rois[:, 2] = 0
            scaled_rois[:, 3] = im_info[1] - 1
            scaled_rois[:, 4] = im_info[0] - 1
        # print 'num fg: {}'.format((labels > 0).sum())
        # print 'num bg: {}'.format((labels == 0).sum())
        if DEBUG and self._iter % 400 == 0:            
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        if self._scale != 0.0 or self._global_context != 0.0:    
            # sampled rois
            top[0].reshape(*rois.shape)
            top[0].data[...] = rois

            top[1].reshape(*scaled_rois.shape)
            top[1].data[...] = scaled_rois

            # classification labels
            top[2].reshape(*labels.shape)
            top[2].data[...] = labels

            # bbox_targets
            top[3].reshape(*bbox_targets.shape)
            top[3].data[...] = bbox_targets

            # bbox_inside_weights
            top[4].reshape(*bbox_inside_weights.shape)
            top[4].data[...] = bbox_inside_weights

            # bbox_outside_weights
            top[5].reshape(*bbox_inside_weights.shape)
            top[5].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)
        else:
            # sampled rois
            top[0].reshape(*rois.shape)
            top[0].data[...] = rois

            # classification labels
            top[1].reshape(*labels.shape)
            top[1].data[...] = labels

            # bbox_targets
            top[2].reshape(*bbox_targets.shape)
            top[2].data[...] = bbox_targets

            # bbox_inside_weights
            top[3].reshape(*bbox_inside_weights.shape)
            top[3].data[...] = bbox_inside_weights

            # bbox_outside_weights
            top[4].reshape(*bbox_inside_weights.shape)
            top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)
                

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _compute_targets(self, ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)
        
        ########my implementation ########
        if DEBUG and self._iter % 400 == 0:
            print "DEBUG for ProposalTargetLayer"
            self._sums += targets[labels != 0, :].sum(axis=0)
            self._squared_sums += (targets[labels != 0, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels != 0)
            # Compute values needed for means and stds
            # var(x) = E(x^2) - E(x)^2
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means and stdenvs for bbox bbox_targets of ProposalTargetLayer!!!!'
            print 'All class means:', means 
            print 'All class stdevs:', stds 

            for i in xrange(self._num_classes-1):
                cls_index = i + 1
                self._cls_sums[i] += targets[labels == cls_index, :].sum(axis=0)  
                self._cls_squared_sums[i] += (targets[labels == cls_index, :] ** 2).sum(axis=0)  
                self._cls_counts[i] += np.sum(labels == cls_index)
                # Compute values needed for means and stds
                # var(x) = E(x^2) - E(x)^2
                cls_means = self._cls_sums[i] / self._cls_counts[i]
                cls_stds = np.sqrt(self._cls_squared_sums[i] / self._cls_counts[i] - cls_means ** 2)
                print 'class %d means:' % (cls_index) 
                print cls_means
                print 'class %d stdevs:' % (cls_index) 
                print cls_stds              
        ########END OF my implementation ########              

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                    / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
        return np.hstack(
                (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

    def _sample_rois(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0].astype(int)
        #print "~~~~~~~~~~~~ number of fore grounds: %d"  % fg_inds.size

        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        # Select sampled values from various arrays:
        labels = labels[keep_inds]
        # Clamp labels for the background RoIs to 0
        labels[fg_rois_per_this_image:] = 0
        rois = all_rois[keep_inds]

        bbox_target_data = self._compute_targets(
            rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, num_classes)

        return labels, rois, bbox_targets, bbox_inside_weights        

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


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights



