import os
import sys
import numpy as np
from conf import *

class Data:
    def __init__(self, file_train, file_test, tuple_format):
        """
        tuple_format specifies the format (ordering) of the data tuple
        e.g.: for a data tuple with 3 input and 2 output,
            [TARGET, TARGET, INPUT, INPUT, INPUT] indicates that target appears prior to input
        """
        def mask_closure(tpl_format, msk_format):
            """
            return closure of mask func
            """
            t_format = tpl_format
            m_key = msk_format
            def mask(d_tuple):
                masked_d = []
                for i in range(len(d_tuple)):
                    if t_format[i] == m_key:
                        masked_d.append(d_tuple[i])
                return masked_d
            return mask

        self.data = np.loadtxt(file_train)
        mask = mask_closure(tuple_format, TARGET)
        self.target = np.array(list(map(mask, self.data)))
        mask = mask_closure(tuple_format, INPUT)
        self.data = np.array(list(map(mask, self.data)))
        # store test data
        self.test_d = np.loadtxt(file_test)
        mask = mask_closure(tuple_format, TARGET)
        self.test_t = np.array(list(map(mask, self.test_d)))
        mask = mask_closure(tuple_format, INPUT)
        self.test_d = np.array(list(map(mask, self.test_d)))

    def __str__(self):
        return '{}data\n{}input is\n{}\ntarget is\n{}\n'.format(line_ddash, line_ddash, self.data, self.target)


class Conf:
    def __init__(self, num_itr, b_rate, w_rate, t_cost):
        self.num_itr = num_itr
        self.b_rate = b_rate
        self.w_rate = w_rate
        # threshold of cost for stop training
        self.t_cost = t_cost
