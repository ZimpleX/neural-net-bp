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
import util.data_proc as data_util
import timeit

import pdb

_TABLE_RAW = 'raw_data|ann'

class Data:
    """
    load training / testing data for neural net
    """
    def _load_db(self, yaml_model, timestamp, profile=True):
        """
        load data from sqlite3 db: suitable for 1D objective function
        """
        db_dir = conf.TRAINING_DIR
        db_name = yaml_model['data_path']
        db_table = yaml_model['data_table']
        db_fullpath = '{}/{}'.format(db_dir, db_name)
        data_size = yaml_model['data_size']
        test_size = yaml_model['test_size']
        start_time = timeit.default_timer()
        conn = sqlite3.connect(db_fullpath)
        c = conn.cursor()
        end_time = timeit.default_timer()
        printf('time spent on db connection: {:.3f}', end_time-start_time)
        # check table exists
        start_time = timeit.default_timer()
        if not db.util.is_table_exist(db_fullpath, db_table, c=c):
            printf('table not exist: {}\npath: {}', db_table, db_fullpath, type='ERROR')
            exit()
        # setup x,y attr name list
        data_attr = list(db.util.get_attr_info(db_table, c=c, enclosing=False).keys())
        regex_x = re.compile('^x\d+$')
        regex_y = re.compile('^y\d+$')
        attr_x = sorted([itm for itm in data_attr if regex_x.match(itm)])
        attr_y = sorted([itm for itm in data_attr if regex_y.match(itm)])
        # load from db
        data_entire = db.util.load_as_array(db_fullpath, db_table, attr_x, size=(data_size+test_size), c=c)
        self.data = data_entire[0:data_size]
        self.test_d = data_entire[-test_size::]
        target_entire = db.util.load_as_array(db_fullpath, db_table, attr_y, size=(data_size+test_size), c=c)
        self.target = target_entire[0:data_size]
        self.test_t = target_entire[-test_size::]
        end_time = timeit.default_timer()
        printf('time spent on load data: {:.3f}', end_time-start_time)
        # store raw into profile db
        prof_subdir = yaml_model['obj_name']
        if profile:
            # don't store y: they will be stored when training starts
            start_time = timeit.default_timer()
            regex_xothers = re.compile('^x.*$')
            attr_xothers = [itm for itm in data_attr if regex_xothers.match(itm) and itm not in attr_x]
            xothers = db.util.load_as_array(db_fullpath, db_table, attr_xothers, size=data_size, c=c)
            attr_full = attr_x + attr_xothers
            data_util.profile_input_data(prof_subdir, timestamp, attr_full, self.data, xothers)
            end_time = timeit.default_timer()
            printf('time spent on storing training data into db: {:.3f}', end_time-start_time)
        conn.close()


    def _load_npz(self, yaml_model):
        """
        load from npz
        NOTE: this is not general loading for all kinds of data.
            It is fitted for digit-classification (USPS) currently
        """
        if yaml_model['data_path'][0] in ['/', '.']:
            data_path = yaml_model['data_path']
        else:
            data_path = '{}/{}'.format(conf.TRAINING_DIR, yaml_model['data_path'])
        data = np.load(data_path)
        batch = yaml_model['batch']
        data_size = (data['train'].shape[0]//batch)*batch
        self.data = data['train'][0:data_size].reshape(-1, yaml_model['input_num_channels'],
            yaml_model['input_image_size_y'], yaml_model['input_image_size_x'])
        self.target = data['train_labels'][0:data_size]
        self.test_d = data['test'].reshape(-1,yaml_model['input_num_channels'],
            yaml_model['input_image_size_y'], yaml_model['input_image_size_x'])
        self.test_t = data['test_labels']
        self.valid_d = data['validation'].reshape(-1,yaml_model['input_num_channels'],
            yaml_model['input_image_size_y'], yaml_model['input_image_size_x'])
        self.valid_t = data['validation_labels']
        yaml_model['data_size'] = self.data.shape[0]
        yaml_model['test_size'] = self.test_d.shape[0]


    def _load_h5(self, yaml_model):
        import tables as tb
        # NOTE: don't forget to close
        self.train_h5 = tb.openFile(yaml_model['h5_train'], mode='r+')
        self.valid_h5 = tb.openFile(yaml_model['h5_valid'], mode='r+')
        self.test_h5 = tb.openFile(yaml_model['h5_test'], mode='r+')
        self.data = self.train_h5.root.data
        self.target = self.train_h5.root.target
        self.valid_d = self.valid_h5.root.data
        self.valid_t = self.valid_h5.root.target
        self.test_d = self.test_h5.root.data
        self.test_t = self.test_h5.root.target
        yaml_model['data_size'] = self.data.shape[0]
        yaml_model['test_size'] = self.test_d.shape[0]


    def _close_h5(self, yaml_model):
        self.train_h5.close()
        self.valid_h5.close()
        self.test_h5.close()

       

    def __init__(self, yaml_model, timestamp, profile=True):
        # 'data_path' means npz contains 'data' / 'validation' / 'test' labels
        if 'data_path' in yaml_model.keys():
            data_type = yaml_model['data_path'].split('.')[-1]
            if data_type == 'db':
                self._load_db(yaml_model, timestamp, profile=profile)
            elif data_type == 'npz':
                self._load_npz(yaml_model)
                self.shuffle = self.shuffle_npz
        elif 'h5_train' in yaml_model.keys():
            self._load_h5(yaml_model)
            self.shuffle = self.shuffle_h5
        else:   # TODO: this is temporary work-around
            assert 'raw_train_dir' in yaml_model.keys()
            assert 'raw_validation_path' in yaml_model.keys()
            self.shuffle = self.shuffle_npz
            import os
            d_tup = ()
            t_tup = ()
            for f in os.listdir(yaml_model['raw_train_dir']):
                # NOTE: no append parent dir
                # NOTE: assume that data is normalized and target is in the right format
                train_raw = np.load('{}/{}'.format(yaml_model['raw_train_dir'],f))
                d_i = train_raw['data'].reshape(-1, yaml_model['input_num_channels'],
                    yaml_model['input_image_size_x'], yaml_model['input_image_size_y'])
                t_i = train_raw['target']
                d_tup += (d_i,)
                t_tup += (t_i,)
            self.data = np.concatenate(d_tup)
            self.target = np.concatenate(t_tup)
            valid_raw = np.load(yaml_model['raw_validation_path'])
            self.valid_d = valid_raw['data'].reshape(-1, yaml_model['input_num_channels'],
                yaml_model['input_image_size_x'], yaml_model['input_image_size_y'])
            self.valid_t = valid_raw['target']
            self.test_d = None
            self.test_t = None
            yaml_model['data_size'] = self.data.shape[0]
            yaml_model['test_size'] = 0
            printf('training data shape: {}', self.data.shape)
            


    def __str__(self):
        return stringf('data') + stringf('input is:\n{}\ntarget is:\n{}', self.data, self.target, type=None, separator=None)

    def get_batches(self, batch_size):
        """
        continuously yield mini-batches of data, 
        by segmenting the whole data set with batch_size
        """
        # if self.data.shape[0] % batch_size != 0:
        #    raise RuntimeError('num of data tuples is not a multiple of batch size')
        from math import ceil
        num_batch = ceil(self.data.shape[0] // batch_size)
        for b in range(num_batch):
            yield self.data[b*batch_size:(b+1)*batch_size, :], \
                    self.target[b*batch_size:(b+1)*batch_size, :]


    def shuffle_npz(self):
        import numpy as np
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.target = self.target[indices]


    def shuffle_h5(self):
        import tables as tb
        import numpy as np
        tot = self.data.shape[0]
        for r in range(tot-1,0,-1):
            idx = np.random.randint(0,r+1)
            _temp = self.data[r]
            self.data[r] = self.data[idx]
            self.data[idx] = _temp
            _temp = self.target[r]
            self.target[r] = self.target[idx]
            self.target[idx] = _temp
