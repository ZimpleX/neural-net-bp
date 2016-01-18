"""
convolution operation: plan to do 2 implementations, 
the efficiency may depend on the image size and net struct
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from node_activity import Node_activity
import pdb


class Node_conv(Node_activity):
    """
    Conv node itself may have non-linear activation functions.
    Define this behavior by y_d_x.
    The main difference from normal node is how you get the input to nodes from w and b
    """
    @classmethod
    def _get_patch(cls, layer, y_start_layer, x_start_layer, kern):
        """
        return a patch of the layer array
    
        ARGUMENTS:
            layer:              (batch) x (channel) x (height) x (width)
            y_start_layer:      starting height index of this patch on [layer]
            x_start_layer:      starting width index of this patch on [layer]
            kern:               kernal size, i.e.: patch width and height
        RETURN:
            (batch) x (channel) x (kern) x (kern)
        """
        batch, channel, Y, X = layer.shape
        patch = np.zeros((batch, channel, kern, kern))
        # clip index when some pixels are in the padding
        y_end_layer = y_start_layer + kern
        x_end_layer = x_start_layer + kern
        y_start_patch = max(0, -y_start_layer)
        x_start_patch = max(0, -x_start_layer)
        y_end_patch = kern - max(0, y_end_layer-Y)
        x_end_patch = kern - max(0, x_end_layer-X)
        y_start_layer = max(0, y_start_layer)
        x_start_layer = max(0, x_start_layer)
        y_end_layer = min(Y, y_end_layer)
        x_end_layer = min(X, x_end_layer)
        patch[..., y_start_patch:y_end_patch, x_start_patch:x_end_patch] = \
            layer[..., y_start_layer:y_end_layer, x_start_layer:x_end_layer]
        return patch

    @classmethod
    def act_forward(cls, prev_layer, w, stride, padding):
        """
        ARGUMENTS:
            prev_layer:     (batch) x (channel_in) x (height) x (width)
            w:              (channel_out) x (channel_in) x (kernel) x (kernel)
            stride:         integer specify stride of w kernel
            padding:        append zero to periphery of prev_layer
        OUTPUT:
            (batch) x (channel_out) x (height') x (width')
            where:
            height' = [height+2*padding-(kernel-1)-1]/stide + 1
                    = (heigth+2*padding-kernel)/stride + 1
            width'  = (width+2*padding-kernel)/stride + 1
        """
        assert prev_layer.shape[1] == w.shape[1]
        batch, chn_in, y_in, x_in = prev_layer.shape
        chn_out, chn_in, kern, kern = w.shape
        y_out = (y_in + 2*padding - kern)//stride + 1
        x_out = (x_in + 2*padding - kern)//stride + 1
        cur_layer = np.zeros((batch, chn_out, y_out, x_out))

        w_flat = w.reshape(chn_out, -1)
        for i in range(y_out):
            y = -padding + i*stride
            for j in range(x_out):
                x = -padding + j*stride
                patch = cls._get_patch(prev_layer, y, x, kern).reshape(batch, -1)
                cur_layer[:, :, i, j] = np.dot(patch, w_flat.T)

        return cur_layer

    @classmethod
    def y_d_x(cls, y_n):
        """
        non-linearity is just clipping
        """
        pass
   
    @classmethod 
    def c_d_w():
        """
        """
        pass

    @classmethod
    def c_d_b(cls, c_d_yn, y_n):
        """
        probably not need to super class version of this
        """
        pass
    
    @classmethod
    def c_d_yn1(cls, c_d_yn, y_n, w):
        pass
