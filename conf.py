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
LEARN_RATE = 0.001
INC_RATE = 1.
DEC_RATE = 1.
MOMENTUM = 0.9
BATCH_SIZE = -1
TRAIN_DATA = TRAINING_DIR + 'Sin_in-1-out-1/08'
TEST_DATA = TRAINING_DIR + 'Sin_in-1-out-1/04'
INIT_RANGE = {'weight': 1,
                'bias': 1}

EPOCH = 500

# specify which column in the data set file stands for target / input
TARGET='TARGET'
INPUT='INPUT'



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
