"""
this module is defining all computation function needed for 
feed forward / back-propagation, for various types of neurons in the whole network

the functions in class definition are almost all classmethod, 
so the class can simply be treated as providing a namespace for the specific neuron type

theoritically, network of arbitrary layers can be constucted by the info in this file,
and be trained recursively using chain rule
"""
import pdb

from abc import ABCMeta, abstractmethod
import numpy as np
from math import exp
import util.array_proc as arr_util
from functools import reduce
import sys


class Node_activity:
    """
    super class for all other node activity classes
    """
    __metaclass__ = ABCMeta
    def __init__():
        pass
    @classmethod
    @abstractmethod
    def act_forward(cls, prev_layer_out, w, b):
        """
        NOTE: overwrite with classmethod

        do the pre-calculation of feed forward provcess on the 
        part that is common to almost all types of neurons

        argument:
        prev_layer_out:     should be 2D, shape[0] is data piece index,
                            shape[1] is num nodes in lower layer
        """
        return np.dot(prev_layer_out, w) + b

    @classmethod
    @abstractmethod
    def y_d_x(cls, y_n):
        """
        NOTE: overwrite with classmethod
        This function is NOT meant to be called directly from outside class.
        It should be wrapped by y_d_b & y_d_w

        ARGUMENT:
        y_n     y list for layer n (batch_size x layer_nodes)
        
        RETURN:
        shape:  the same as input (batch_size x layer_nodes)
        """
        return np.ones(y_n.shape)

    @classmethod
    def y_d_w(cls, y_n, y_n_1):
        """
        NOTE: DON'T overwrite this function when creating subclass

        get derivative w.r.t. the whole weight matrix in one layer

        argument:
        y_n     y list of layer n
        y_n_1   y list of layer n-1 (prev layer)

        --> return matrix shape is .. x .. x (nodes in n-1 layer) x (nodes in n layer)
            e.g.: return[..., i, j] = d(y_nj) / d(w_(n-1)ij)
        """
        # expansion on y_n_1
        d_chain = arr_util.expand_col_swap(y_n_1, y_n.shape[-1])
        d_layer = cls.y_d_x(y_n)
        d_layer = arr_util.expand_col(d_layer, y_n_1.shape[-1])
        # apply chain rule
        return d_chain * d_layer

    @classmethod
    def y_d_b(cls, y_n):
        """
        NOTE: DON'T overwrite this function when creating subclass
        get derivative w.r.t. the whole bias vector in one layer

        argument:
        y_n     y list for layer n (batch_size x layer_nodes)

        return:
        shape: the same as input (batch_size x layer_nodes)
        """
        return cls.y_d_x(y_n)

    @classmethod
    def yn_d_yn1(cls, y_n, w):
        """
        NOTE: DON'T overwrite this function when creating subclass

        w is the weight array, which should be strictly 2D
        return the derivative of y_n w.r.t y_(n-1)
        return[...,i,j] = d(y_nj) / d(y_n1i)
        """
        assert len(w.shape) == 2
        # shape w/o the last dimension
        hi_dim_shp = [y_n.shape[i] for i in range(len(y_n.shape)-1)]
        exp_coef = reduce(lambda x, y: x*y, hi_dim_shp)
        d_chain = np.expand_dims(w, axis=0)
        d_chain = np.repeat(d_chain, exp_coef, axis=0)
        shape = hi_dim_shp + list(w.shape)
        d_chain = d_chain.reshape(shape)

        d_layer = cls.y_d_x(y_n)
        d_layer = arr_util.expand_col(d_layer, w.shape[0])
        return d_chain * d_layer



class Node_linear(Node_activity):
    """
    linear neuron
    """
    @classmethod
    def act_forward(cls, prev_layer_out, w, b):
        """
        feed forward linear neuron
        """
        return super(Node_linear, cls).act_forward(prev_layer_out, w, b)
    
    @classmethod
    def y_d_x(cls, y_n):
        """
        linear derivative
        """
        return np.ones(y_n.shape)


class Node_sigmoid(Node_activity):
    """
    sigmoid (logistic) neuron
    """
    @classmethod
    def act_forward(cls, prev_layer_out, w, b):
        """
        feed forward sigmoid neuron
        """
        z = super(Node_sigmoid, cls).act_forward(prev_layer_out, w, b)
        shape = z.shape
        sz = z.size
        # flatten z and compute exp, then restore original shape
        # python3.x: have to convert to list first
        # check -300: exp(300) will cause MathOverflow
        expz = list(map(lambda x: (x>-300) and exp(-x) or sys.float_info.max, 
                        z.reshape(sz)))
        expz = np.array(expz)
        expz = expz.reshape(shape)
        return 1. / (1 + expz)

    @classmethod
    def y_d_x(cls, y_n):
        """
        sigmoid derivative, expressed in terms of y (NOT x)
        """
        return y_n * (1 - y_n)



#########################################
#########################################
activation_dict = {'sig': Node_sigmoid, 
                   'lin': Node_linear}
