"""
basic building block for db manipulation:
*   populate data
*   remove data (sanity)
"""
import os
import sqlite3
from time import strftime
import logf.filef as filef
from logf.printf import *
from functools import reduce
from numpy import *     # array()

import pdb

_DB_DIR_PARENT = './profile_data/'
_DB_NAME = 'ann.db'
_DB_TABLE = 'null|null' # format compatible with Benchtracker



def populate_db(attr_name, attr_type, *d_tuple, db_path=_DB_DIR_PARENT, db_name=_DB_NAME, 
        table_name=_DB_TABLE, append_time=True, usr_time=None, silent=False):
    """
    populate data into database, with user defined schema
    optionally append the time to each data tuple.
    Policy on existing data:
        This function will never drop existing data. It will only append to 
        table if it already exists. So this is a safe operation. 
        To delete entries, call sanity_db. 
    
    ARGUMENTS:
        attr_name       list of attribute name in database
        attr_type       type of attr: e.g.: INTEGER, TEXT, REAL...
        d_tuple         arbitrary num of arguments that consist of the tuple
                        can be 1D or 2D: if 1D, expand it to 2D
        db_path         path of database
        db_name         name of database
        table_name      table name
        append_time     append timestamp to each data tuple if set true
        silent          won't log info after successful population if set True
    """
    file_opt = 'a'
    db_fullname = '{}/{}'.format(db_path, db_name)
    filef.mkdir_r(os.path.dirname(db_fullname))
    open(db_fullname, file_opt).close()
    # set-up
    num_tuples = -1
    d_fulltuple = None
    for d in d_tuple:
        d_arr = array(d)
        assert len(d_arr.shape) <= 2
        if len(d_arr.shape) == 2:
            num_tuples = (num_tuples > -1) and num_tuples or d_arr.shape[0]
            assert num_tuples == d_arr.shape[0]
    if num_tuples == -1:    # only one tuple
        num_tuples = 1
    for d in d_tuple:
        d_arr = array(d)
        if len(d_arr.shape) == 0:
            d_arr = array([d])
        if len(d_arr.shape) == 1:
            d_arr = d_arr.reshape(1, d_arr.size)
            d_arr = repeat(d_arr, num_tuples, axis=0)
        if d_fulltuple is None:
            d_fulltuple = d_arr
        else:
            d_fulltuple = concatenate((d_fulltuple, d_arr), axis=1)

    if append_time:
        attr_name = ['populate_time'] + list(attr_name)
        attr_type = ['TEXT'] + list(attr_type)
        time_str = usr_time and usr_time or strftime('[%D-%H:%M:%S]')
        time_col = array([time_str] * num_tuples) \
                .reshape(num_tuples, 1)
        d_fulltuple = concatenate((time_col, d_fulltuple), axis=1)
    # sqlite3
    conn = sqlite3.connect(db_fullname)
    c = conn.cursor()
    table_name = '[{}]'.format(table_name)
    assert len(attr_name) == len(attr_type)
    create_clause = ['[{}] {}'.format(attr_name[i], attr_type[i]) \
                    for i in range(len(attr_name))]
    create_clause = reduce(lambda a,b: '{}, {}'.format(a, b), create_clause)
    create_clause = 'CREATE TABLE IF NOT EXISTS {} ({})'.format(table_name, create_clause)
    c.execute(create_clause)
    for tpl in d_fulltuple:
        tpl_str = ['?'] * len(tpl)
        tpl_str = reduce(lambda a,b:'{}, {}'.format(a, b), tpl_str)
        insert_clause = 'INSERT INTO {} VALUES ({})'.format(table_name, tpl_str)
        c.execute(insert_clause, tpl)
    # finish up
    conn.commit()
    conn.close()
    # log
    if not silent:
        printf('success: populate {} entries into table {}', 
                num_tuples, table_name, separator=None)



def sanity_db(attr_name, attr_val, table_name, db_name=_DB_NAME, db_path=_DB_DIR_PARENT, silent=False):
    """
    remove entries in db file. Can be useful to keep the db clean
    when you do a lot of unstable testing for ANN.

    ARGUMENTS:
        attr_name       list: the selection criteria for deleting db entries
        attr_val        list: the selection value for deleting db entries
        table           the table in the db file
        db_name
        db_path         the full path of db file is db_name + db_path
        silent          if silent, don't log info after successful deletion
    """
    # convert arg to list if passing in single int / string
    if len(array(attr_name).shape) == 0:
        attr_name = [attr_name]
    if len(array(attr_val).shape) == 0:
        attr_val = [attr_val]
    db_fullname = '{}/{}'.format(db_path, db_name)
    # don't check file: leave it to user / wrapper function
    conn = sqlite3.connect(db_fullname)
    c = conn.cursor()
    table_name = '[{}]'.format(table_name)
    orig_row = c.execute('SELECT Count(*) FROM {}'.format(table_name)).fetchone()[0]
    attr_len = len(attr_name)
    assert attr_len == len(attr_val)
    attr_val = list(map(lambda s: (type(s)==type('')) and '\'{}\''.format(s) or s, attr_val))
    del_cond = ['[{}] = {}'.format(attr_name[i], attr_val[i]) for i in range(attr_len)]
    del_cond = reduce(lambda a,b: '{} and {}'.format(a,b), del_cond)
    c.execute('DELETE FROM {} WHERE {}'.format(table_name, del_cond))
    fina_row = c.execute('SELECT Count(*) FROM {}'.format(table_name)).fetchone()[0]
    conn.commit()
    conn.close()
    if not silent:
        printf('success: delete {} entries from {}', orig_row-fina_row, table_name, type='WARN')
