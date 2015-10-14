# some constant definition
TARGET=0
INPUT=1

line_dash=('-' * 20) + '\n'
line_ddash=('=' * 20) + '\n'
line_star=('*' * 20) + '\n'

class Conf:
    def __init__(self, num_epoch, rate, t_cost):
        self.num_epoch = num_epoch
        self.b_rate = rate
        self.w_rate = rate
        # threshold of cost for stop training
        self.t_cost = t_cost
