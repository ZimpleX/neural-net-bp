"""
convolution operation: plan to do 2 implementations, 
the efficiency may depend on the image size and net struct
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from node_activity import Node_activity
import pdb

from fractions import Fraction, gcd
from math import ceil, floor

class Node_conv(Node_activity):
    """
    Conv node itself may have non-linear activation functions.
    Define this behavior by y_d_x.
    The main difference from normal node is how you get the input to nodes from w and b
    
    [Design choice]: 
    It is arguable that I should not make this sub-class of Node_activity, cuz there is
    not really anything to reuse from super-class. 
    And in this sense, it may even be better to change the following from class method to 
    normal method, and store kern size & padding & stride as member variable.
    For now, I prefer not to do so, considering the consistency in main training body:
    maintaining current implementation simplify the logic in main body.

    [NOTE]:
    make sure that the kernel width is equal to kernel height!!
    """
    @classmethod
    def _get_patch(cls, base_mat, y_start_base, x_start_base, dy, dx, unit):
        """
        return a patch of the base matrix
        [NOTE]: y_start_base, x_start_base can be fractional number; dx, dy is counted in `unit`
                i.e.: this `unit` is the patch_stride
        ARGUMENTS:
            base_mat:           (batch) x (channel) x (height) x (width)
            y_start_base:       starting height index of this patch on [base_mat]
            x_start_base:       starting width index of this patch on [base_mat]
            dy, dx:             kernal size, i.e.: patch width and height
            unit:               increment index by `unit` a time: can be fractional
                                if index is non-integer, then the value is zero
        RETURN:
            (batch) x (channel) x (dy) x (dx)
        """
        batch, channel, Y, X = base_mat.shape
        patch = np.zeros((batch, channel, dy, dx))
        # base mat: index setup
        stride_base  = max(unit, 1)
        y_start_idx = max(ceil(y_start_base), 0)
        x_start_idx = max(ceil(x_start_base), 0)
        y_end_idx = min(ceil(y_start_base + dy*unit), Y)
        x_end_idx = min(ceil(x_start_base + dx*unit), X)
        patch_fill = base_mat[..., y_start_idx:y_end_idx:stride_base, x_start_idx:x_end_idx:stride_base]
        # patch: index setup
        stride_patch = max(1/unit, 1)
        num_x, num_y = patch_fill[-2::]
        unit = (unit>1) and 1 or unit
        y_patch_start = (max(ceil(y_start_base), 0) - y_start_base) / unit
        x_patch_start = (max(ceil(x_start_base), 0) - x_start_base) / unit
        y_patch_end = (num_y-1)*stride_patch + 1 + y_patch_start
        x_patch_end = (num_x-1)*stride_patch + 1 + x_patch_start
        # fill in
        patch[..., y_patch_start:y_patch_end:stride_patch, \
                    x_patch_start:x_patch_end:stride_patch] = patch_fill
        return patch


    @classmethod
    def _conv4dflip(cls, base_mat, kern_mat, sliding_stride, patch_stride, padding):
        """
        Convolution method ONLY for 4d numpy array
        Operation: ret_mat = base_mat (*) flipped(kern_mat).
        [NOTE1]: don't swap the position of base_mat & kern_mat: padding is added to base_mat
        [NOTE2]: it won't matter here if `kern_mat.shape[-1] != kern_mat.shape[-2]`,
            HOWEVER, if that is the case, other places may be broken.
        [NOTE3]: padding & patch_stride & sliding_stride can all be fractional
        The shape of input & output:
            [A x b x c x d] (*) [E x b x f x g] = [A x E x m x n]
            (where assuming c > f; d > g)
            *   dimension A, E are retained
            *   dimension b is eliminated
            *   dimension m = (c + 2*padding - 1 - (f-1)*patch_stride) / sliding_stride + 1
            *   dimension n = (d + 2*padding - 1 - (g-1)*patch_stride) / sliding_stride + 1
        """
        assert base_mat.shape[1] == kern_mat.shape[1]
        A, b, c, d = base_mat.shape
        E, b, f, g = kern_mat.shape
        unit = min(1, gcd(gcd(base_mat, kern_mat), padding))
        m = (c + 2*padding - 1 - (f-1)*patch_stride)/sliding_stride + 1
        n = (c + 2*padding - 1 - (g-1)*patch_stride)/sliding_stride + 1
        ret_mat = np.zeros((A, E, m, n))
        kern_mat_flat = kern_mat.reshape(E, -1)
        for i in range(m):
            y = -padding + i*sliding_stride
            for j in range(n):
                x = -padding + j*sliding_stride
                patch = cls._get_patch(base_mat, y, x, f, g, patch_stride).reshape(A, -1)
                ret_mat[:,:,i,j] = np.dot(patch, kern_mat_flat.T)
        return ret_mat
        


    @classmethod
    def act_forward(cls, prev_layer, w, stride, padding):
        """
        NOTE:
            w is actually the flipped kernel:
            y = x (*) kernel = x (*) flipped(w)
        ARGUMENTS:
            prev_layer:     (batch) x (channel_in) x (height) x (width)
            w:              (channel_in) x (channel_out) x (kernel) x (kernel)
            stride:         integer specify stride of w kernel
            padding:        append zero to periphery of prev_layer
        OUTPUT:
            (batch) x (channel_out) x (height') x (width')
            please refer to _conv4dflip.
        """
        # TODO: may want non-linear activation: ReU
        """
        Feed forward:
            padding: integer
            sliding_stride: >= 1, integer
            patch_stride: = 1
        """
        return cls._conv4dflip(prev_layer, np.swapaxes(w, 0, 1), stride, padding)

    @classmethod
    def y_d_x(cls, y_n):
        """
        non-linearity is just clipping
        """
        pass
   
    @classmethod 
    def c_d_w(cls, c_d_yn, y_n, y_n_1, stride, padding):
        """
        c_d_w = y_n_1 (*) flipped(c_d_yn x yn_d_x)
        """
        """
        patch_stride = S
        sliding_stride = 1
        padding = p
        """

        c_d_xn = c_d_yn * cls.y_d_x(y_n)

        pass

    @classmethod
    def c_d_yn1(cls, c_d_yn, y_n, w, stride, padding):
        """
        c_d_yn1 = (c_d_yn x yn_d_x) (*) w
        [NOTE1]: things get tricky when stride not equal to 1:
            basically you have to get a patch with holes
        [NOTE2]: padding passed in is the feed-forward padding, not the padding 
            used for conv in this function
        """
        c_d_xn = c_d_yn * cls.y_d_x(y_n)
        assert w.shape[-1] == w.shape[-2]   # kernel must be square
        padding2 = w.shape[-1] - padding - 1
        """
        padding = (k-p-1)/S
        patch_stride = 1/S
        sliding_stride = 1/S
        """
        return cls._conv4dflip(c_d_xn, w[:,:,::-1,::-1], 1, padding2, patch_stride=stride)
