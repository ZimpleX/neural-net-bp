"""
set up data (training / testing) for training the ANN
"""
import sqlite3
from logf.printf import *
from logf.stringf import *
import numpy as np
import conf

import pdb

class Data:
    def __init__(self, data_size, test_size, table_name, 
        db_dir=conf.TRAINING_DIR, db_data=conf.DB_DATA, db_test=conf.DB_TEST):
        """
        load training data and test data from table_name
        """
        data_fullpath = '{}/{}'.format(db_dir, db_data)
        test_fullpath = '{}/{}'.format(db_dir, db_test)
        conn = sqlite3.connect(data_fullpath)
        c = conn.cursor()
