"""
set up data (training / testing) for training the ANN
"""
from logf.printf import *
from logf.stringf import *
import conf
import db_util as db
import re
import sqlite3
import timeit

import pdb

class Data:
    def __init__(self, data_size, test_size, table_data, table_test,
        db_dir=conf.TRAINING_DIR, db_data=conf.DB_DATA, db_test=conf.DB_TEST):
        """
        load training data and test data from table_name
        """
        data_fullpath = '{}/{}'.format(db_dir, db_data)
        test_fullpath = '{}/{}'.format(db_dir, db_test)
        # check table exists
        start_time = timeit.default_timer()
        conn_d = sqlite3.connect(data_fullpath)
        c_d = conn_d.cursor()
        conn_t = sqlite3.connect(test_fullpath)
        c_t = conn_t.cursor()
        end_time = timeit.default_timer()
        printf('time spent on db connection: {}', end_time-start_time)
        start_time = timeit.default_timer()
        if not db.util.is_table_exist(data_fullpath, table_data, c=c_d) or \
            not db.util.is_table_exist(test_fullpath, table_test, c=c_t):
            printf('table not exist: {} or {} \npath: {} or {}', table_data, table_test, data_fullpath, test_fullpath, type='ERROR')
            exit()
        # setup x,y attr name list
        data_attr = list(db.util.get_attr_info(data_fullpath, table_data, c=c_d, enclosing=False).keys())
        test_attr = list(db.util.get_attr_info(test_fullpath, table_test, c=c_t, enclosing=False).keys())
        regex_x = re.compile('^x\d+$')
        regex_y = re.compile('^y\d+$')
        data_attr_x = [itm for itm in data_attr if regex_x.match(itm)]
        data_attr_y = [itm for itm in data_attr if regex_y.match(itm)]
        test_attr_x = [itm for itm in test_attr if regex_x.match(itm)]
        test_attr_y = [itm for itm in test_attr if regex_y.match(itm)]
        # load
        self.data = db.util.load_as_array(data_fullpath, table_data, data_attr_x, size=pow(2, data_size), c=c_d)
        self.target = db.util.load_as_array(data_fullpath, table_data, data_attr_y, size=pow(2, data_size), c=c_d)
        self.test_d = db.util.load_as_array(test_fullpath, table_test, test_attr_x, size=pow(2, test_size), c=c_t)
        self.test_t = db.util.load_as_array(test_fullpath, table_test, test_attr_y, size=pow(2, test_size), c=c_t)
        end_time = timeit.default_timer()
        printf('time spent on load data: {}', end_time-start_time)
        conn_d.close()
        conn_t.close()

    def __str__(self):
        return stringf('data') + stringf('input is:\n{}\ntarget is:\n{}', self.data, self.target, type=None, separator=None)

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

