from abc import ABCMeta, abstractmethod
import numpy as np
from math import exp

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
    def y_d_w(cls, y_n, y_n_1, w_idx):
        """
        NOTE: overwrite with classmethod

        derivative of y w.r.t. w, where y is the output of 
        the neuron, and w is the weight connecting y and y 
        from previous layer.
        do the pre-calc for the part that is common for all sub-class

        argument:
        y_n     y list for nth layer
        y_n_1   y list for (n-1) layer (prev layer)
        w_idx   index tuple of weight (i.e. w is connecting 
                which 2 neurons)
        """
        # the last dimension of y_n should be num of nodes in nth layer
        deriv = np.zeros(y_n.shape)
        deriv[..., w_idx[1]] = y_n_1[..., w_idx[0]]
        return deriv

    @classmethod
    @abstractmethod
    def y_d_b(cls, y_n, b_idx):
        """
        NOTE: overwrite with classmethod

        derivative w.r.t. bias

        argument:
        y_n     y list for nth layer
        b_idx   bias index, suggesting which node's bias is under concern
        """
        deriv = np.zeros(y_n.shape)
        deriv[..., b_idx] = 1
        return deriv


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
    def y_d_w(cls, y_n, y_n_1, w_idx):
        return super(Node_linear, cls).y_d_w(y_n, y_n_1, w_idx)
    
    @classmethod
    def y_d_b(cls, y_n, b_idx):
        return super(Node_linear, cls).y_d_b(y_n, b_idx)


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
        expz = list(map(lambda x: exp(x), z.reshape(sz)))
        expz = np.array(expz)
        expz = expz.reshape(shape)
        return 1. / (1 + expz)

    @classmethod
    def y_d_w(cls, y_n, y_n_1, w_idx):
        y_n_sub = y_n[..., w_idx[1]]
        d_sigmo = y_n_sub * (1 - y_n_sub)
        d_chain = super(Node_sigmoid, cls).y_d_w(y_n, y_n_1, w_idx)[..., w_idx[1]]
        deriv = np.zeros(y_n.shape)
        deriv[..., w_idx[1]] = d_sigmo * d_chain
        return deriv

    @classmethod
    def y_d_b(cls, y_n, b_idx):
        y_n_sub = y_n[..., b_idx]
        d_sigmo = y_n_sub * (1 - y_n_sub)
        d_chain = super(Node_sigmoid, cls).y_d_b(y_n, b_idx)[..., b_idx]
        deriv = np.zeros(y_n.shape)
        deriv[..., b_idx] = d_sigmo * d_chain
        return deriv
