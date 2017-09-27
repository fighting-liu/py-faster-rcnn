import caffe
import numpy as np


class SigmoidCrossEntropyWithWeight(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.wpos = np.ones((1, 191))*2
        self.wneg = np.ones((1, 191))  
        self.eps=1e-6
        self.eps1 = 1 + self.eps                  

    def reshape(self, bottom, top):
        # print '~~~~~~~~~~~~~~~~~~~'
        #print bottom[0].data.shape
        #print bottom[1].data.shape
        # check input dimensions match
	print bottom[0].count, bottom[1].count
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.prob = np.zeros_like(bottom[0].data, dtype=np.float32)


        # self.wpos = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape((1, -1))  ## 1*6
        # elf.wneg = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape((1, -1))  ## 1*6

        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
        ## default: pre=bottom[0]   gt_label=bottom[1]
        self.prob = vector_simoid(bottom[0].data) ## m*6
        gt_label = np.squeeze(bottom[1].data)
        top[0].data[...] = -np.sum((gt_label * vector_log(self.prob+self.eps) * self.wpos + (1-gt_label) * vector_log(self.eps1-self.prob) * self.wneg)) / bottom[0].num        
        # top[0].data[...] = -np.sum((bottom[1].data * vector_log(self.prob+self.eps) * self.wpos + (1-bottom[1].data) * vector_log(self.eps1-self.prob) * self.wneg)) / bottom[0].num
        # print top[0].data[...]


    def backward(self, top, propagate_down, bottom):
        gt_label = np.squeeze(bottom[1].data)
        for i in range(2):
            if not propagate_down[i]:
                continue
            else:
                bottom[i].diff[...] = -(gt_label*self.wpos - self.prob*self.wneg + gt_label*self.prob*(self.wneg-self.wpos)) / bottom[0].num 
                # bottom[i].diff[...] = -(bottom[1].data*self.wpos - self.prob*self.wneg + bottom[1].data*self.prob(self.wneg-self.wpos)) / bottom[0].num   

def vector_log(x):
    return np.log(x)

def vector_simoid(x):
    ## sigmoid computation
    return 1.0 / (1+np.exp(-x))
