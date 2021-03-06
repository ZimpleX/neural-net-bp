"""
default configuration for the ANN: 
    can be overwritten by cmd line args
"""
TRAINING_DIR = './train_data/'
DB_DIR_PARENT = './profile_data/'

STAT = {'conv_time': 0.}

class Conf:
    """
    storing all training configuration (which is independent of net configuration)
    """
    def __init__(self, yaml_model):
        self.num_epoch = yaml_model['epoch']
        self.batch = yaml_model['batch']
        self.b_rate = yaml_model['alpha']
        self.w_rate = yaml_model['alpha']
        self.momentum = yaml_model['momentum']

    def __str__(self):
        return  """
                training parameters:
                --------------------
                learning rate (bias):   {}
                learning rate (weight): {}
                momentum:               {}
                """.format(self.b_rate, self.w_rate, self.momentum)


#########################################
#    Currently supported activations    #
#########################################
from net.node_activity import *
from conv.conv_layer import *
from conv.pool_layer import *

activation_dict = { 'FC_SIG': Node_sigmoid,
                    'FC_LIN': Node_linear,
                    'FC_SOFTMAX': Node_softmax,
                    'CONVOLUTION': Node_conv,
                    'MAXPOOL': Node_pool}

##################################
#    Currently supported cost    #
##################################
from net.cost import *

cost_dict = {'SQR': Cost_sqr,
             'CE': Cost_ce}
