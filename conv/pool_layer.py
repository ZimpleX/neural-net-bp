"""
pooling layer to be operated together with conv layer.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from node_activity import Node_activity
from conv.conv_layer import Node_conv
import conv.slid_win as slid
from fractions import Fraction


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
        batch, channel = prev_layer.shape[0:2]
        pseudo_kern = np.zeros((batch, channel, self.kern, self.kern))
        return slid.slid_win_4d_flip(prev_layer, pseudo_kern, self.stride, 1, self.padding, slid.pool_ff())
    

    @classmethod
    def y_d_x(cls, y_n): pass


    def c_d_w_b_yn1(self, c_d_yn, y_n, y_n_1):
        """
        NO derivative of w and b
        """
        batch, channel = prev_layer.shape[0:2]
        pseudo_kern = np.zeros((batch, channel, self.kern, self.kern))
        yn_zip = np.concatenate((y_n,c_d_yn), axis=1)    # double the channel
        pd = Fraction(self.kern - self.padding - 1, self.stride)
        ps = ss = Fraction(1, self.stride)
        c_d_yn1 = slid.slid_win_4d_flip(yn_zip, pseudo_kern, ss, ps, pd, slid.pool_bp(y_n_1))
        return None, None, c_d_yn1
