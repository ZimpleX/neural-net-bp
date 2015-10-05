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
    def y_d_w_single(cls, y_n, y_n_1, w_idx):
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
    def y_d_w_mat(cls, y_n, y_n_1):
        # TODO: change this to get deriv w.r.t. every w in weight matrix directly
        dim = len(y_n.shape)    # num of dimension of output matrix
        n_node = y_n.shape[-1]
        n_1_node = y_n_1.shape[-1]
        shape = list(y_n_1.shape)
        shape.insert(len(shape)-1, n_node)
        exp_y_n_1 = np.repeat(y_n_1, n_node, axis=dim-2) \
                    .reshape(shape).swapaxes(dim-1, dim-2)
        return exp_y_n_1

    @classmethod
    @abstractmethod
    def y_d_b_single(cls, y_n, b_idx):
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

    @classmethod
    @abstractmethod
    def y_d_b_mat(cls, y_n):
        pass


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
    def y_d_w_single(cls, y_n, y_n_1, w_idx):
        """
        probably obsolete: use mat version
        """
        return super(Node_linear, cls).y_d_w_single(y_n, y_n_1, w_idx)

    @classmethod
    def y_d_w_mat(cls, y_n, y_n_1):
        return super(Node_linear, cls).y_d_w_mat(y_n, y_n_1)
    
    @classmethod
    def y_d_b_single(cls, y_n, b_idx):
        """
        probably obsolete: use mat version
        """
        return super(Node_linear, cls).y_d_b_single(y_n, b_idx)


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
    def y_d_w_single(cls, y_n, y_n_1, w_idx):
        """
        probably obsolete: use mat version
        """
        y_n_sub = y_n[..., w_idx[1]]
        d_sigmo = y_n_sub * (1 - y_n_sub)
        d_chain = super(Node_sigmoid, cls).y_d_w_single(y_n, y_n_1, w_idx)[..., w_idx[1]]
        deriv = np.zeros(y_n.shape)
        deriv[..., w_idx[1]] = d_sigmo * d_chain
        return deriv

    @classmethod
    def y_d_w_mat(cls, y_n, y_n_1, w_idx):
        dim = len(y_n.shape)
        n_node = y_n.shape[-1]
        n_1_node = y_n_1.shape[-1]
        d_chain = super(Node_sigmoid, cls).y_d_w_mat(y_n, y_n_1)
        d_sigmo = y_n * (1 - y_n)
        shape = list(y_n.shape)
        shape.insert(len(shape)-1, n_1_node)
        d_sigmo = np.repeat(d_sigmo, n_1_node, axis=dim-2).reshape(shape)
        return d_chain * d_sigmo

    @classmethod
    def y_d_b_single(cls, y_n, b_idx):
        """
        probably obsolete: use mat version
        """
        y_n_sub = y_n[..., b_idx]
        d_sigmo = y_n_sub * (1 - y_n_sub)
        d_chain = super(Node_sigmoid, cls).y_d_b_single(y_n, b_idx)[..., b_idx]
        deriv = np.zeros(y_n.shape)
        deriv[..., b_idx] = d_sigmo * d_chain
        return deriv

