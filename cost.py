from abc import ABCMeta, abstractmethod
import numpy as np
from math import exp

class Cost:
    __metaclass__ = ABCMeta
    @classmethod
    @abstractmethod
    def act_forward(cls):
        pass

    @classmethod
    @abstractmethod
    def c_d_y(cls):
        """
        derivative of cost w.r.t. output
        """
        pass


class Cost_sqr(Cost):
    """
    square sum cost function
    """
    @classmethod
    def act_forward(cls, y_out, t):
        return 0.5 * sum((y_out-t)*(y_out-t), axis=-1)
