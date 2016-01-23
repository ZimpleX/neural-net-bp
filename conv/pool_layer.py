"""
pooling layer to be operated together with conv layer.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from node_activity import Node_activity
from conv.conv_layer import Node_conv
import conv.convolution as conv


class Node_pool(Node_conv):
    """
    Pooling layer: feed-forward and backward gradient can be viewed as 
    a simplified version of convolutional layer.

    Pooling layer doesn't have anything to learn.
    """
    __metaclass__ = ABCMeta
    def __init__(self, kern, stride, padding):
        self.kern = kern
        self.stride = stride
        self.padding = padding

    def act_forward(self, prev_layer):
        """
        input and output have the same number of channels
        """
        batch, channel, height_in, width_in = prev_layer.shape
        height_out = (height_in + 2*self.padding - self.kern) / self.stride + 1
        width_out  = (width_in + 2*self.padding - self.kern) / self.stride + 1
        ret_mat = np.zeros((batch, channel, height_out, width_out))
        for i in range(height_out):
            y = -self.padding + i*self.stride
            for j in range(width_out):
                x = -self.padding + j*self.stride
                patch = conv.get_patch(prev_layer, y, x, self.kern, self.kern, 1)
                ret_mat[:,:,i,j] = patch.reshape(batch, channel, -1).max(axis=2)
        return ret_mat
    
    @classmethod
    def y_d_x(cls, y_n):
        pass

    def c_d_yn1(self, c_d_yn, y_n, y_n_1):
        """
        NO derivative of w and b
        """
        batch, channel, height_n1, width_n1 = y_n_1.shape
        c_d_yn1 = np.zeros(y_n_1.shape)
        for i in range(c_d_yn.shape[-2]):
            y = -self.padding + i*self.stride
            for j in range(c_d_yn.shape[-1]):
                x = -self.padding + j*self.stride
                patch = conv.get_patch(y_n_1, y, x, self.kern, self.kern, self.stride)
                max_flt = y_n[:,:,i,j,np.newaxis,np.newaxis]    # w/o new axis, it is 2D
                deriv_flt = c_d_yn[:,:,i,j,np.newaxis,np.newaxis]
                deriv_patch = (patch == max_flt) * deriv_flt    # broadcast rule
                conv.update_patch(c_d_yn1, deriv_patch, y, x)
        return c_d_yn1
                
