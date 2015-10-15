# conf for the ANN: 
#   can be overwritten by cmd line arg
STRUCT = [3, 9, 1]
ACTIVATION = ['sig', 'lin']
COST = 'sqr'
LEARN_RATE = 0.01
INC_RATE = 1.
DEC_RATE = 1.
MOMENTUM = 0.
BATCH_SIZE = 1
TRAIN_DATA = 'train_data/AttenSin3/3_08'
TEST_DATA = 'train_data/AttenSin3/3_04'

EPOCH = 100

# specify which column in the data set file stands for target / input
TARGET='TARGET'
INPUT='INPUT'
# utils for printing
line_dash=('-' * 20) + '\n'
line_ddash=('=' * 20) + '\n'
line_star=('*' * 20) + '\n'




class Conf:
    def __init__(self, num_epoch, rate, inc_rate, dec_rate, momentum, t_cost):
        self.num_epoch = num_epoch
        self.b_rate = rate
        self.w_rate = rate
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.momentum = momentum
        # threshold of cost for stop training
        self.t_cost = t_cost
