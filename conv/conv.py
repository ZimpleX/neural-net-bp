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
    def _get_patch(cls, layer, y_start_layer, x_start_layer, dy, dx, stride):
        """
        return a patch of the layer array
    
        ARGUMENTS:
            layer:              (batch) x (channel) x (height) x (width)
            y_start_layer:      starting height index of this patch on [layer]
            x_start_layer:      starting width index of this patch on [layer]
            dy, dx:             kernal size, i.e.: patch width and height
            stride:             used in bp of conv layer --> if sliding window act_forward has [stride != 1]
        RETURN:
            (batch) x (channel) x (dy) x (dx)
        """
        # TODO: add support for stride
        batch, channel, Y, X = layer.shape
        patch = np.zeros((batch, channel, dy, dx))
        # clip index when some pixels are in the padding
        y_end_layer = y_start_layer + dy
        x_end_layer = x_start_layer + dx
        y_start_patch = max(0, -y_start_layer)
        x_start_patch = max(0, -x_start_layer)
        y_end_patch = dy - max(0, y_end_layer-Y)
        x_end_patch = dx - max(0, x_end_layer-X)
        y_start_layer = max(0, y_start_layer)
        x_start_layer = max(0, x_start_layer)
        y_end_layer = min(Y, y_end_layer)
        x_end_layer = min(X, x_end_layer)
        patch[..., y_start_patch:y_end_patch, x_start_patch:x_end_patch] = \
            layer[..., y_start_layer:y_end_layer, x_start_layer:x_end_layer]
        return patch


    @classmethod
    def _conv4dflip(cls, base_mat, sliding_mat, stride, padding):
        """
        Convolution method for 4d numpy array array
        I won't optimize this for other dimensions as they won't appear in DCNN.
        Operation is straight-forward: ret_mat = base_mat (*) flipped(sliding_mat).
        [NOTE1]: don't swap the position of base_mat & sliding_mat: padding is added to base_mat
        [NOTE2]: it won't matter here if `sliding_mat.shape[-1] != sliding_mat.shape[-2]`,
            HOWEVER, if that is the case, other places may be broken.
        The shape of input & output:
            [A x b x c x d] (*) [E x b x f x g] = [A x E x m x n]
            (where assuming c > f; d > g)
            *   dimension A, E are retained
            *   dimension b is eliminated
            *   dimension m = [c + 2*padding - (f-1) - 1] / stride + 1
                            = (c + 2*padding - f) / stride + 1
            *   dimension n = [d + 2*padding - (g-1) - 1] / stride + 1
                            = (d + 2*padding - g) / stride + 1
        """
        assert base_mat.shape[1] == sliding_mat.shape[1]
        A, b, c, d = base_mat.shape
        E, b, f, g = sliding_mat.shape
        m = (c + 2*padding - f)//stride + 1 # TODO: probably want more strict: '/' instead of '//'
        n = (c + 2*padding - g)//stride + 1
        ret_mat = np.zeros((A, E, m, n))
        sliding_mat_flat = sliding_mat.reshape(E, -1)
        for i in range(m):
            y = -padding + i*stride
            for j in range(n):
                x = -padding + j*stride
                patch = cls._get_patch(base_mat, y, x, f, g).reshape(A, -1)
                ret_mat[:,:,i,j] = np.dot(patch, sliding_mat_flat.T)
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
        return cls._conv4dflip(c_d_xn, w[:,:,::-1,::-1], stride, padding2)
