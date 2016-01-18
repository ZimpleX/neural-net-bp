"""
default configuration for the ANN: 
    can be overwritten by cmd line arg
"""
STRUCT = [3, 15, 1]
ACTIVATION = ['sig', 'lin']
COST = 'sqr'
TRAINING_DIR = './train_data/'
DB_DATA = 'data.db'
DB_TEST = 'test.db'
TABLE_DATA = 'sin_is-3-os-1-ir-(-10~10)-or-(-5~5)|ann'
TABLE_TEST = 'sin_is-3-os-1-ir-(-10~10)-or-(-5~5)|ann'
SIZE_DATA = 8
SIZE_TEST = 4
LEARN_RATE = 0.04
INC_RATE = 1.
DEC_RATE = 1.
MOMENTUM = 0.9
BATCH_SIZE = -1
INIT_RANGE = {'weight': 1,
                'bias': 1}

EPOCH = 1000



class Conf:
    """
    storing all training configuration (which is independent of net configuration)
    """

    def __init__(self, num_epoch, rate, inc_rate, dec_rate, momentum, t_cost):
        self.num_epoch = num_epoch
        self.b_rate = rate
        self.w_rate = rate
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.momentum = momentum
        # threshold of cost for stop training
        self.t_cost = t_cost

    def __str__(self):
        return  """
                training parameters:
                --------------------
                learning rate (bias):   {}
                learning rate (weight): {}
                incremental rate:       {}
                decremental rate:       {}
                momentum:               {}
                """.format(self.b_rate, self.w_rate, 
                        self.inc_rate, self.dec_rate,
                        self.momentum)


#########################################
#    Currently supported activations    #
#########################################
from node_activity import *
from conv.conv import *

activation_dict = { 'sig': Node_sigmoid,
                    'lin': Node_linear,
                    'softmax': Node_softmax,
                    'conv': Node_conv}
