"""
convolution operation: plan to do 2 implementations, 
the efficiency may depend on the image size and net struct
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from node_activity import Node_activity
#import conv.slide_win as slid
import conv.slide_win_spark as slid
import pdb

from fractions import Fraction


class Node_conv(Node_activity):
    """
    Convolutional layer: both the forward-acting and backward gradient can be expressed by conv.
    
    [NOTE]:
    make sure that the kernel width is equal to kernel height!!
    """
    __metaclass__ = ABCMeta
    def __init__(self, stride, padding):
        self.stride = stride
        self.padding = padding


    def act_forward(self, prev_layer, w, b):
        """
        NOTE:
            w is actually the flipped kernel:
            y = x (*) kernel = x (*) flipped(w)
        ARGUMENTS:
            prev_layer:     (batch) x (channel_in) x (height) x (width)
            w:              (channel_in) x (channel_out) x (kernel) x (kernel)
            b:              (channel_out)
        OUTPUT:
            (batch) x (channel_out) x (height') x (width')
            please refer to slid_win_4d_flip for height' and width'
        """
        ret = slid.slid_win_4d_flip(prev_layer, np.swapaxes(w, 0, 1), 
                self.stride, 1, self.padding, slid.convolution())
        b_exp = b[np.newaxis, :, np.newaxis, np.newaxis]
        return np.clip(ret+b_exp, 0, np.finfo(np.float64).max)    # ReLU

    @classmethod
    def y_d_x(cls, y_n):
        """
        ReLU: non-linearity is just clipping
        """
        return np.greater(y_n, 0.)
   

    def c_d_w_b_yn1(self, c_d_yn, y_n, y_n_1, w, is_c_d_yn1=1):
        """
        stride, padding are both the integer value in the feed forward case.
        get derivative of cost w.r.t. weight, bias, prev_layer.

        ARGUMENT:
            c_d_yn      (batch) x (channel_n) x (height) x (width)
            y_n         (batch) x (channel_n) x (height) x (width)
            y_n_1       (batch) x (channel_n_1) x (height') x (width')
            w           (channel_n_1) x (channel_n) x (kern) x (kern)
        """
        c_d_xn = self._c_d_xn(c_d_yn, y_n)
        c_d_b = np.sum(c_d_xn, axis=(0,2,3))
        ####  c_d_w  ####
        ##  y_n_1 (*) flipped(c_d_xn)
        #   patch_stride = stride
        #   padding = padding
        #   slide_stride = 1
        c_d_w = slid.slid_win_4d_flip(np.swapaxes(y_n_1,0,1), np.swapaxes(c_d_xn,0,1), 
                1, self.stride, self.padding, slid.convolution())
        assert c_d_w.shape == w.shape
        ####  c_d_yn1  ####
        ##  c_d_xn (*) w ##
        #   patch_stride = 1/stride
        #   padding = (kern-padding-1)/stride
        #   slide_stride = 1/stride
        pad2 = Fraction(w.shape[-1] - self.padding - 1, self.stride)
        c_d_yn1 = slid.slid_win_4d_flip(c_d_xn, w[:,:,::-1,::-1], Fraction(1, self.stride), 
                Fraction(1, self.stride), pad2, slid.convolution())
        assert c_d_yn1.shape == y_n_1.shape
        return c_d_w, c_d_b, c_d_yn1
