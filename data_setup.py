import os
import sys
import numpy as np
from conf import *

class Data:
    def __init__(self, file_train, file_test):
        """
        Read in from the text file "file_train" & "file_test".
        The text files should contain only float numbers, each line representing
        one single training datum (target + input).
        Within the same directory of the text file, there should be a conf file,
        specifying the format (ordering) of the data tuple
            e.g.: for a data tuple with 3 input and 2 output,
                [TARGET, TARGET, INPUT, INPUT, INPUT] indicates that target appears prior to input
        
        NOTE:
        It is recommended that you use the `util/data_generator.py` to generate
        the training text file (conf will be dealt with automatically).
        """
        def mask_closure(tpl_format, msk_format):
            """
            return closure of mask func
            """
            t_format = tpl_format
            m_key = msk_format
            def mask(d_tuple):
                """
                mask data according to tuple format & mask format.
                e.g.:
                    [TARGET, INPUT, TARGET, INPUT, INPUT] & TARGET
                    ==> return the 1st and 3rd data in the tuple
                """
                masked_d = []
                for i in range(len(d_tuple)):
                    if t_format[i] == m_key:
                        masked_d.append(d_tuple[i])
                return masked_d
            return mask

        # setup tuple format
        conf_file = file_train.split('/')
        if len(conf_file) == 1:
            conf_file = './conf'
        else:
            conf_file[-1] = 'conf'  # the whole list except last element
            conf_file = '/'.join(conf_file)
        fconf = open(conf_file, 'r')
        tuple_format = fconf.readline().split()
        fconf.close()

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

    def get_batches(self, batch_size):
        """
        continuously yield mini-batches of data, 
        by segmenting the whole data set with batch_size
        """
        batch_size = (batch_size==-1) and self.data.shape[0] or batch_size
        if self.data.shape[0] % batch_size != 0:
            raise RuntimeError('num of data tuples is not a multiple of batch size')
        num_batch = self.data.shape[0] // batch_size
        for b in range(num_batch):
            yield self.data[b*batch_size:(b+1)*batch_size, :], \
                    self.target[b*batch_size:(b+1)*batch_size, :]

