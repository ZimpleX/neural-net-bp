from abc import ABCMeta, abstractmethod
import numpy as np

tiny = 1e-30    # avoid math error for log()

#####################
#  cost superclass  #
#####################
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


###################
#  cost subclass  #
###################
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
        return (y_out - t) / y_out.shape[0]


class Cost_ce(Cost):
    """
    cross-entropy cost function.
    [NOTE]: this MUST be used together with softmax, because c_d_y will actually
        compute c_d_x directly. --> See doc in 'node_activity.Node_softmax' for explanation.
    """
    @classmethod
    def act_forward(cls, y_out, t):
        return -np.sum(t * np.log(y_out + tiny), axis=-1)

    @classmethod
    def c_d_y(cls, y_out, t):
        """
        [NOTE] AGAIN: this is NOT c_d_y. This is c_d_x.
        """
        return (y_out - t) / y_out.shape[0]


