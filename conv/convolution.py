"""
abstract out the convolution operation from the conv_layer.
"""

import numpy as np
from math import ceil, floor
from fractions import gcd


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
    stride_base  = max(unit, 1)
    y_start_idx = max(ceil(y_start_base), 0)
    x_start_idx = max(ceil(x_start_base), 0)
    y_end_idx = min(ceil(y_start_base + dy*unit), Y)
    x_end_idx = min(ceil(x_start_base + dx*unit), X)
    patch_fill = base_mat[..., y_start_idx:y_end_idx:stride_base, x_start_idx:x_end_idx:stride_base]
    # patch: index setup
    stride_patch = max(1/unit, 1)
    num_x, num_y = patch_fill[-2::]
    unit = min(1, unit)
    y_patch_start = (max(ceil(y_start_base), 0) - y_start_base) / unit
    x_patch_start = (max(ceil(x_start_base), 0) - x_start_base) / unit
    y_patch_end = (num_y-1)*stride_patch + y_patch_start + 1
    x_patch_end = (num_x-1)*stride_patch + x_patch_start + 1
    # fill in
    patch[..., y_patch_start:y_patch_end:stride_patch, \
                x_patch_start:x_patch_end:stride_patch] = patch_fill
    return patch


def conv4dflip(base_mat, kern_mat, sliding_stride, patch_stride, padding):
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
            patch = get_patch(base_mat, y, x, f, g, patch_stride).reshape(A, -1)
            ret_mat[:,:,i,j] = np.dot(patch, kern_mat_flat.T)
    return ret_mat


