"""
abstract out the convolution operation from the conv_layer.

OBSERVATION:
baseline version: 
    convolution     85%
    get_patch       30%
    dot_product     50%
"""

import numpy as np
from math import ceil, floor
from abc import ABCMeta, abstractmethod
import timeit
from stat_cnn.time import RUNTIME

import pdb



def get_patch(base_mat, y_start_base, x_start_base, dy, dx, unit):
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
    stride_base  = int(max(unit, 1))
    y_start_idx = max(ceil(y_start_base), int(y_start_base%unit))
    x_start_idx = max(ceil(x_start_base), int(x_start_base%unit))
    y_end_idx = min(ceil(y_start_base + dy*unit), Y)
    x_end_idx = min(ceil(x_start_base + dx*unit), X)
    patch_fill = base_mat[..., y_start_idx:y_end_idx:stride_base, x_start_idx:x_end_idx:stride_base]
    # patch: index setup
    stride_patch = int(max(1/unit, 1))
    num_y, num_x = patch_fill.shape[-2::]
    y_patch_start = ceil((max(ceil(y_start_base), 0) - y_start_base) / unit)
    x_patch_start = ceil((max(ceil(x_start_base), 0) - x_start_base) / unit)
    y_patch_end = int((num_y-1)*stride_patch + y_patch_start + 1)
    x_patch_end = int((num_x-1)*stride_patch + x_patch_start + 1)
    # fill in
    patch[..., y_patch_start:y_patch_end:stride_patch, \
                x_patch_start:x_patch_end:stride_patch] = patch_fill
    return patch


def slid_win_4d_flip(base_mat, kern_mat, sliding_stride, patch_stride, padding, func_obj):
    """
    Method ONLY for 4d numpy array
    Operation: slide kern_mat through base_mat, according to stride and padding.
        func_obj        object specifying (patch,kern) operation, preprocessing, and additional args
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
    m = (c + 2*padding - 1 - (f-1)*patch_stride)/sliding_stride + 1
    n = (d + 2*padding - 1 - (g-1)*patch_stride)/sliding_stride + 1
    ret_mat = np.zeros((A, E, int(m), int(n)))
    func_obj.pre_proc(base_mat, kern_mat)
    y = -padding - sliding_stride
    # double for loop is a map function
    start_time = timeit.default_timer()
    import ec2.sc_glob as spark
    t4 = timeit.default_timer()
    s = 2
    rdd_base_exp = spark.sc.parallelize([None]*s*s).map(lambda x: base_mat)
    t5 = timeit.default_timer()
    RUNTIME['repeat_base_mat'] += t5 - t4
    mm = int(m)
    nn = int(n)
    for i in range(0, mm, s):
        for j in range(0, nn, s):
            # MapReduce here
            for ii in range(i, min(i+s,mm)):
                for jj in range(j , min(j+s,nn)):
                    y = -padding + ii*sliding_stride
                    x = -padding + jj*sliding_stride
                    t1 = timeit.default_timer()
                    patch = get_patch(base_mat, y, x, f, g, patch_stride)
                    t2 = timeit.default_timer()
                    ret_mat[:,:,i,j] = func_obj.patch_func(patch,i,j)
                    t3 = timeit.default_timer()
                    RUNTIME['get_path_time'] += t2 - t1
                    RUNTIME['dot_time'] += t3 - t2
    end_time = timeit.default_timer()
    RUNTIME['conv_time'] += end_time - start_time
    return ret_mat


###########################
#  operations on a patch  #
###########################
class slide_operation:
    """
    protocal for all sliding-window type of processing functions
    """
    __metaclass__ = ABCMeta
    def __init__():
        pass
    @abstractmethod
    def pre_proc(self, base, kern):
        pass
    @abstractmethod
    def patch_func(self, patch, i, j):
        pass



class convolution(slide_operation):
    def __init__(self): pass
        
    def pre_proc(self, base, kern):
        """
        this is to avoid the reshape operation inside the double for loop.
        Python won't automatically optimize it for you.
        """
        self.A = base.shape[0]
        E = kern.shape[0]
        self.kern_trans = kern.reshape(E, -1).T
    def patch_func(self, patch, i, j):
        """
        operation for convolution
        """
        return np.dot(patch.reshape(self.A,-1), self.kern_trans)


class pool_ff(slide_operation):
    """
    feed-forward of pooling layer
    """
    def __init__(self): pass

    def pre_proc(self, base, kern): pass

    def patch_func(self, patch, i, j):
        return patch.max(axis=(-1,-2))


class pool_bp(slide_operation):
    """
    back-prop of pooling layer
    """
    def __init__(self, y_n_1):
        self.y_n_1 = y_n_1

    def pre_proc(self, base, kern):
        self.channel = base.shape[1]/2
    
    def patch_func(self, patch, i, j):
        y_n_patch = patch[:,0:self.channel,:,:]
        c_d_yn_patch = patch[:,self.channel::,:,:]
        y_n1_flt = self.y_n_1[:,:,i,j,np.newaxis,np.newaxis]
        return np.sum((y_n1_flt == y_n_patch) * c_d_yn_patch, axis=(-1,-2))
