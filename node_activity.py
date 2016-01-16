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


#########################################
#    super class for all activations    #
#########################################
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

        ARGUMENT:
            prev_layer_out:     (batch) x (N_n_1)
            w:                  weight, (N_n_1) x (N_n)
            b:                  bias, N_n
        RETURN:
            (batch) x (N_n)
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
            y_n     y list for layer n: (batch) x (N_n)
        RETURN:
            (batch) x (N_n)
        """
        return np.ones(y_n.shape)

    @classmethod
    def c_d_w(cls, c_d_yn, y_n, y_n_1):
        """
        NOTE: DON'T overwrite this function when creating subclass

        get derivative of cost w.r.t. weight between layer n and n-1

        ARGUMENT:
            c_d_yn  derivative of cost w.r.t. layer n output:   (batch) x (N_n)
            y_n     y list of layer n (cur layer):              (batch) x (N_n)
            y_n_1   y list of layer n-1 (prev layer):           (batch) x (N_n_1)
        RETURN:
            weight derivative of shape:     (N_n_1) x (N_n)
            the return has already summed over all batch entries
        """
        d_chain = y_n_1                     # (batch) x (N_n_1)
        c_d_xn = c_d_yn * cls.y_d_x(y_n)    # (batch) x (N_n)
        return np.dot(d_chain.T, c_d_xn)    # dot product is summing over mini-batch

    @classmethod
    def c_d_b(cls, c_d_yn, y_n):
        """
        NOTE: DON'T overwrite this function when creating subclass
        get derivative of cost w.r.t. bias vector in layer n

        ARGUMENT:
            c_d_yn  derivative of cost w.r.t. layer n output    (batch) x (N_n)
            y_n     y list for layer n:                         (batch) x (N_n)
        RETURN:
            (N_n)
        """
        return np.sum((c_d_yn * cls.y_d_x(y_n)), axis=0)

    @classmethod
    def c_d_yn1(cls, c_d_yn, y_n, w):
        """
        NOTE: DON'T overwrite this function when creating subclass
        get derivative of cost w.r.t. output of layer n-1

        ARGUMENT:
            c_d_yn  derivative of cost w.r.t. layer n output    (batch) x (N_n)
            y_n     output of layer n                           (batch) x (N_n)
            w       weight between layer n and n-1              (N_n_1) x (N_n)
        RETURN
            (batch) x (N_n_1)
        """
        yn_d_x = cls.y_d_x(y_n)
        return np.dot((c_d_yn * yn_d_x), w.T)



##############################
#    specific activations    #
##############################
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
        expz = super(Node_sigmoid, cls).act_forward(prev_layer_out, w, b)
        expz = np.clip(expz, -400, np.finfo(np.float64).max) # prevent exp overflow
        expz = np.exp(-expz)    # elementwise exp on ndarray
        return 1. / (1 + expz)

    @classmethod
    def y_d_x(cls, y_n):
        """
        sigmoid derivative, expressed in terms of y (NOT x)
        """
        return y_n * (1 - y_n)


class Node_softmax(Node_activity):
    """
    softmax (logit) layer: node i value depends on all nodes on the same layer.
    suitable for classification.
        y = exp(x) / [E + exp(x)]
    """
    @classmethod
    def act_forward(cls, prev_layer_out, w, b):
        """
        feed forward softmax
        """
        expz = super(Node_softmax, cls).act_forward(prev_layer_out, w, b)
        expz = np.clip(expz, np.finfo(np.float64).min, 400) # prevent exp overflow
        expz = np.exp(expz)     # elementwise exp on ndarray
        # denominator: for normalizing output as a probability distribution
        deno_z = np.sum(expz, axis=-1).reshape(expz.shape[0],1)
        return expz / deno_z  # refer to numpy broadcasting rule
    
    @classmethod
    def y_d_x(cls, y_n):
        """
        softmax derivative
        """
        return y_n * (1 - y_n)
        


#########################################
#    currently supported activations    #
#########################################
activation_dict = { 'sig': Node_sigmoid, 
                    'lin': Node_linear,
                    'softmax': Node_softmax}
