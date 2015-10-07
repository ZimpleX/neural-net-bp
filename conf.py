# some constant definition
TARGET=0
INPUT=1

line_dash=('-' * 20) + '\n'
line_ddash=('=' * 20) + '\n'
line_star=('*' * 20) + '\n'

class Conf:
    def __init__(self, num_itr, b_rate, w_rate, t_cost):
        self.num_itr = num_itr
        self.b_rate = b_rate
        self.w_rate = w_rate
        # threshold of cost for stop training
        self.t_cost = t_cost
