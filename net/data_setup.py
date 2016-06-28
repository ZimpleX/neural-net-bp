"""
set up data (training / testing) for training the ANN
"""
from logf.printf import *
from logf.stringf import *
import net.conf as conf
import db_util as db
import re
import sqlite3
import numpy as np
import timeit

import pdb

class Data:
    """
    load training / testing data for neural net
    """
    def __init__(self, yaml_model, timestamp):
        """
        supported file extensions:
            *.h5
            *.npz
        """
        import os
        self.all_data = {'train': [None,None],  # (data, target)
                         'valid': [None,None],
                         'test':  [None,None]}
        data_dir = yaml_model['data_dir']
        f_list = [i for i in os.listdir(data_dir) if i[0] != '.']
        f_format = yaml_model['data_format']
        assert set(f_list) == set(['{}.{}'.format(i,f_format) for i in self.all_data.keys()])
        self.f_opened = []
        if f_format == 'h5':
            self._load_h5(f_list, data_dir, yaml_model)
            self.shuffle = self.shuffle_h5
            self.cleanup = self.cleanup_h5
        elif f_format == 'npz':
            self._load_npz(f_list, data_dir, yaml_model)
            self.shuffle = self.shuffle_npz
            self.cleanup = self.cleanup_npz
        else:
            printf('invalid file format: {}.\t\texit!',f_format)
            exit()

           
    def _load_npz(self, f_list, data_dir, yaml_model):
        """
        load from npz
        NOTE: this is not general loading for all kinds of data.
            It is fitted for digit-classification (USPS) currently
        """
        for f in f_list:
            npzf = np.load('{}/{}'.format(data_dir,f))
            f_info = f.split('.')
            assert len(f_info) == 2
            assert f_info[1] == 'npz'
            self.all_data[f_info[0]] = [npzf['data'], npzf['target']]
            yaml_model['{}_size'.format(f_info[0])] = npzf['target'].shape[0]


    def _load_h5(self, f_list, data_dir, yaml_model):
        import tables as tb
        for f in f_list:
            # NOTE: don't forget to close
            h5f = tb.openFile('{}/{}'.format(data_dir,f), mode='r+')
            self.f_opened += [h5f]
            f_info = f.split('.')
            assert len(f_info) == 2
            assert f_info[1] == 'h5'
            self.all_data[f_info[0]] = [h5f.root.data, h5f.root.target]
            yaml_model['{}_size'.format(f_info[0])] = h5f.root.target.shape[0]


    def cleanup_npz(self): pass

    def cleanup_h5(self):
        for h5f in self.f_opened:
            h5f.close()



    def __str__(self):
        return stringf('data') + stringf('input is:\n{}\ntarget is:\n{}', \
            self.all_data['train'][0], self.all_data['train'][1], type=None, separator=None)


    def shuffle_npz(self):
        import numpy as np
        d_key = 'train'
        indices = np.arange(self.all_data[d_key][1].shape[0])
        np.random.shuffle(indices)
        self.all_data[d_key][0] = self.all_data[d_key][0][indices]
        self.all_data[d_key][1] = self.all_data[d_key][1][indices]


    def shuffle_h5(self):
        import tables as tb
        import numpy as np
        d_key = 'train'
        tot = self.all_data[d_key][1].shape[0]
        for r in range(tot-1,0,-1):
            idx = np.random.randint(0,r+1)
            _temp = self.all_data[d_key][0][r]
            self.all_data[d_key][0][r] = self.all_data[d_key][0][idx]
            self.all_data[d_key][0][idx] = _temp
            _temp = self.all_data[d_key][1][r]
            self.all_data[d_key][1][r] = self.all_data[d_key][1][idx]
            self.all_data[d_key][1][idx] = _temp
