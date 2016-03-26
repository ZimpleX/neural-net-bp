"""
pooling layer to be operated together with conv layer.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from net.node_activity import Node_activity
from conv.conv_layer import Node_conv
import conv.slide_win as slide_serial
import conv.slide_win_spark as slide_spark
from fractions import Fraction
from functools import reduce

import pdb


class Node_pool(Node_conv):
    """
    Pooling layer: feed-forward and backward gradient can be viewed as 
    a simplified version of convolutional layer.

    Pooling layer doesn't have anything to learn.
    """
    __metaclass__ = ABCMeta
    def __init__(self, chan, kern, stride, padding, slid_method, SparkMeta):
        self.chan = chan
        self.kern = kern
        self.stride = stride
        self.padding = padding
        self.slid_win_4d_flip = eval(slid_method).slid_win_4d_flip
        self.pool_ff = eval(slid_method).pool_ff
        self.pool_bp = eval(slid_method).pool_bp
        self.SparkMeta = SparkMeta


    def act_forward(self, prev_layer, _, __, sc=None):
        """
        input and output have the same number of channels
        """
        ret = self.slid_win_4d_flip(prev_layer, {'channel':self.chan,'kern':self.kern}, self.stride, 1, self.padding, self.pool_ff(), sc, SparkMeta=self.SparkMeta)
        if sc is not None and self.SparkMeta['conn_to_FC']:
            # ret = ret.collect()
            ret = ret.reduce(lambda _1,_2: np.concatenate((_1,_2),axis=0))
        return ret
                
    

    @classmethod
    def y_d_x(cls, y_n): pass


    def c_d_w_b_yn1(self, c_d_yn, y_n, y_n_1, _, is_c_d_yn1=None, sc=None):
        """
        NO derivative of w and b
        """
        batch, channel = y_n_1.shape[0:2]
        pseudo_kern = np.zeros((channel, channel*2, self.kern, self.kern))
        yn_zip = np.concatenate((y_n,c_d_yn.reshape(y_n.shape)), axis=1)    # double the channel
        pd = Fraction(self.kern - self.padding - 1, self.stride)
        ps = ss = Fraction(1, self.stride)
        c_d_yn1 = self.slid_win_4d_flip(yn_zip, pseudo_kern, ss, ps, pd, self.pool_bp(y_n_1), sc, SparkMeta=self.SparkMeta)
        return None, None, c_d_yn1
