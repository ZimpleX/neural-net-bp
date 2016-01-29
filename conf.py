"""
default configuration for the ANN: 
    can be overwritten by cmd line arg
"""
TRAINING_DIR = './train_data/'
DB_DIR_PARENT = './profile_data/'


class Conf:
    """
    storing all training configuration (which is independent of net configuration)
    """
    def __init__(self, yaml_model):
        self.num_epoch = yaml_model['epoch']
        self.batch = yaml_model['batch']
        self.b_rate = yaml_model['alpha']
        self.w_rate = yaml_model['alpha']
        self.inc_rate = yaml_model['inc_rate']
        self.dec_rate = yaml_model['dec_rate']
        self.momentum = yaml_model['momentum']

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
from cost import *

cost_dict = {'SQR': Cost_sqr,
                'CE': Cost_ce}
