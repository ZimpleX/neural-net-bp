"""
this file defines handy array manipulation that is common
in network training
"""

import numpy as np

def expand_col(arr, repeat_num):
    """
    repeat the data in the second last dimension,
    make it a new dimension
    """
    dim = len(arr.shape)
    shape = list(arr.shape)
    shape.insert(len(shape)-1, repeat_num)
    return np.repeat(arr, repeat_num, axis=dim-2).reshape(shape)


def expand_col_swap(arr, repeat_num):
    """
    repeat the data in the second last dimension,
    make it a new dimension and swap the last 2 axes
    """
    dim = len(arr.shape)
    return expand_col(arr, repeat_num).swapaxes(dim-1, dim-2)
