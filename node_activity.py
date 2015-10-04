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
    @abstractmethod
    def act_forward(cls, prev_layer_out, w, b):
        """
        NOTE: this should be overriden with classmethod
        """
        pass
    @classmethod
    def pre_process(cls, prev_layer_out, w, b):
        """
        do the pre-calculation of the part that is common
        for almost all type of neurons
        """
        return np.dot(prev_layer_out, w) + b


class Node_linear(Node_activity):
    """
    linear neuron
    """
    @classmethod
    def act_forward(cls, prev_layer_out, w, b):
        """
        feed forward linear neuron
        """
        return super(Node_linear, cls).pre_process(prev_layer_out, w, b)


class Node_sigmoid(Node_activity):
    """
    sigmoid (logistic) neuron
    """
    @classmethod
    def act_forward(cls, prev_layer_out, w, b):
        """
        feed forward sigmoid neuron
        """
        z = super(Node_sigmoid, cls).pre_process(prev_layer_out, w, b)
        shape = z.shape
        size = z.size
        # flatten z and compute exp, then restore original shape
        expz = np.array(map(lambda x: exp(x), z.reshape(size)))
        expz = expz.reshape(shape)
        return 1. / (1 + expz)
