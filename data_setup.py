"""
set up data (training / testing) for training the ANN
"""
from logf.printf import *
from logf.stringf import *
import conf
import db_util as db
import re
import sqlite3
import util.data_proc as data_util
import timeit

import pdb

_TABLE_RAW = 'raw_data|ann'

class Data:
    """
    load training / testing data for neural net
    """
    def _load_db(self, yaml_model, timestamp, profile=True):
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
        printf('time spent on db connection: {}', end_time-start_time)
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
        printf('time spent on load data: {}', end_time-start_time)
        # store raw into profile db
        prof_subdir = ''
        if profile:
            # don't store y: they will be stored when training starts
            start_time = timeit.default_timer()
            regex_xothers = re.compile('^x.*$')
            attr_xothers = [itm for itm in data_attr if regex_xothers.match(itm) and itm not in attr_x]
            xothers = db.util.load_as_array(db_fullpath, db_table, attr_xothers, size=data_size, c=c)
            attr_full = attr_x + attr_xothers
            data_util.profile_input_data(prof_subdir, timestamp, attr_full, self.data, xothers)
            end_time = timeit.default_timer()
            printf('time spent on storing training data into db: {}', end_time-start_time)
        conn.close()


        
    def __init__(self, yaml_model, timestamp, profile=True):
        data_type = yaml_model['data_path'].split('.')[-1]
        if data_type == 'db':
            self._load_db(yaml_model, timestamp, profile=profile)
        # else:
        #   ...


    def __str__(self):
        return stringf('data') + stringf('input is:\n{}\ntarget is:\n{}', self.data, self.target, type=None, separator=None)

    def get_batches(self, batch_size):
        """
        continuously yield mini-batches of data, 
        by segmenting the whole data set with batch_size
        """
        if self.data.shape[0] % batch_size != 0:
            raise RuntimeError('num of data tuples is not a multiple of batch size')
        num_batch = self.data.shape[0] // batch_size
        for b in range(num_batch):
            yield self.data[b*batch_size:(b+1)*batch_size, :], \
                    self.target[b*batch_size:(b+1)*batch_size, :]

