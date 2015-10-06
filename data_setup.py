import os
import sys
import numpy as np
from conf import *

class Data:
    def __init__(self, file_name, tuple_format):
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

        self.data = np.loadtxt(file_name)
        mask = mask_closure(tuple_format, TARGET)
        self.target = np.array(list(map(mask, self.data)))
        mask = mask_closure(tuple_format, INPUT)
        self.data = np.array(list(map(mask, self.data)))

    def __str__(self):
        return 'input is\n{}\ntarget is\n{}\n'.format(self.data, self.target)
