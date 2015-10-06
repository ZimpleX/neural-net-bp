from abc import ABCMeta, abstractmethod
import numpy as np
from math import exp

class Cost:
    __metaclass__ = ABCMeta
    @classmethod
    @abstractmethod
    def act_forward(cls):
        """
        NOTE: overwrite this with classmethod
        """
        pass

    @classmethod
    @abstractmethod
    def c_d_y(cls):
        """
        NOTE: overwrite this with classmethod

        derivative of cost w.r.t. output layer
        """
        pass


class Cost_sqr(Cost):
    """
    square sum cost function
    """
    @classmethod
    def act_forward(cls, y_out, t):
        return 0.5 * np.sum((y_out-t)*(y_out-t), axis=-1)

    @classmethod
    def c_d_y(cls, y_out, t):
        """
        return a derivative matrix
        """
        return y_out - t
