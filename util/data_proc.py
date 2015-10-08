"""
e.g.: populate data in db
"""
import os
import sqlite3
import sys
from np import *
from time import strftime


def populate_db(attr_name, attr_type, db_path='../data/', db_name='unnamed.db', 
        table_name='null|null', overwrite=False, append_time=True, *d_tuple):
    """
    populate data into database
    optionally append the time to each data tuple
    
    arguments:
    attr_name       list of attribute name in database
    attr_type       type of attr: e.g.: INTEGER, TEXT, REAL...
    db_path         path of database
    db_name         name of database
    table_name      table name
    overwrite       erase data in existing db if set true
    append_time     append timestamp to each data tuple if set true
    d_tuple         arbitrary num of arguments that consist of the tuple
                    can be 1D or 2D: if 1D, expand it to 2D
    """
    if not os.path.isdir(db_path):
        os.makedirs(db_path)
    #file_opt = overwrite and 'w' or 'a'
    file_opt = 'a'
    db_fullname = db_path + '/' + db_name
    open(db_fullname, db_fullname).close()
    # set-up
    num_tuples = -1
    d_fulltuple = None
    for d in d_tuple:
        d_arr = array(d)
        assert len(d_arr.shape) <= 2
        if len(d_arr.shape) == 2:
            num_tuples = (num_tuples > -1) and num_tuples or d_arr.shape[1]
            assert num_tuples == d_arr.shape[1]
    if num_tuples == -1:    # only one tuple
        num_tuples = 1
    for d in d_tuple:
        d_arr = array(d)
        if len(d_arr.shape) == 1:
            d_arr = d_arr.reshape(1, d_arr.size)
            d_arr = repeat(d_arr, num_tuples, axis=0)
        if d_fulltuple == None:
            d_fulltuple = d_arr
        else:
            concatenate((d_fulltuple, d_arr), axis=1)

    if append_time:
        attr_name = ['populate_time'] + list(attr_name)
        attr_type = ['TEXT'] + list(attr_type)
        time_col = array([strftime('%Y-%M-%D %H:%M:%S')] * num_tuples) \
                .reshape(num_tuples, 1)
        concatenate((time_col, d_fulltuple), axis=1)
    # sqlite3
    conn = sqlite3.connect(db_fullname)
    c = conn.cursor()
    if overwrite:
        c.execute('DROP TABLE IF EXISTS {}'.format(table_name))
    assert len(attr_name) == len(attr_type)
    create_clause = ['{} {}'.format(attr_name, attr_type) \
                    for i in range(len(attr_name))]
    create_clause = reduce(lambda a,b: '{}, {}'.format(a, b), create_clause)
    create_clause = 'CREATE TABLE IF NOT EXISTS {} ({})'.format(table_name, create_clause)
    c.execute(create_clause)
    for tpl in d_fulltuple:
        tpl_str = reduce(lambda a,b: '{}, {}'.format(a,b), tpl)
        insert_clause = 'INSERT INTO {} VALUES ({})'.format(table_name, tpl_str)
        c.execute(insert_clause)

    conn.commit()
    conn.close()
