"""
e.g.: populate data in db
"""
import os
import sqlite3
import sys
from numpy import *
from time import strftime
from functools import reduce
import pdb


def populate_db(attr_name, attr_type, *d_tuple, db_path='profile_data/', db_name='ann.db', 
        table_name='null|null', overwrite=False, append_time=True, usr_time=None):
    """
    populate data into database
    optionally append the time to each data tuple
    
    arguments:
    attr_name       list of attribute name in database
    attr_type       type of attr: e.g.: INTEGER, TEXT, REAL...
    d_tuple         arbitrary num of arguments that consist of the tuple
                    can be 1D or 2D: if 1D, expand it to 2D
    db_path         path of database
    db_name         name of database
    table_name      table name
    overwrite       erase data in existing db if set true
    append_time     append timestamp to each data tuple if set true
    """
    if not os.path.isdir(db_path):
        os.makedirs(db_path)
    #file_opt = overwrite and 'w' or 'a'
    file_opt = 'a'
    db_fullname = db_path + '/' + db_name
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
    if overwrite:
        c.execute('DROP TABLE IF EXISTS {}'.format(table_name))
    assert len(attr_name) == len(attr_type)
    create_clause = ['{} {}'.format(attr_name[i], attr_type[i]) \
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



def profile_net_conf(args, timestamp):
    """
    populate the conf of the ANN, timestamp is to identify each run:
    for join of different tables in the future
    """
    struct_info = [str(args.struct[i+1])+args.activation[i] for i in range(len(args.activation))]
    net_struct = '{}lin'.format(args.struct[0])
    net_struct += '-' + reduce(lambda a,b:a+'-'+b, struct_info)

    net_attr = ['struct', 'cost_type', 'train_data_name', 'test_data_name', 
            'batch_size', 'learn_rate', 'inc_rate', 'dec_rate', 'momentum']
    net_val = array([net_struct, args.cost, args.path_train, args.path_test, 
            args.batch, args.rate, args.inc_rate, args.dec_rate, args.momentum])
    net_attr_type = ['TEXT', 'TEXT', 'TEXT', 'TEXT', 
            'INTEGER', 'REAL', 'REAL', 'REAL', 'REAL']
    populate_db(net_attr, net_attr_type, net_val, 
            table_name='meta|ann', usr_time=timestamp)


def profile_raw_data_set(data_set, timestamp):
    """
    populate the raw data set that is used to train ANN
    """
    data_attr = ['x{}'.format(i) for i in range(data_set.data.shape[1])]
    data_attr += ['y{}'.format(i) for i in range(data_set.target.shape[1])]
    data_attr_type = ['REAL'] * (data_set.data.shape[1] + data_set.target.shape[1])
    populate_db(data_attr, data_attr_type, data_set.data, 
            data_set.target, table_name='raw_data|ann', usr_time=timestamp)



def profile_cost(epoch, batch, cost_bat, timestamp):
    """
    cost_bat is the sum of cost over some batches
    """
    prof_attr = ['epoch_num', 'batch_num', 'cost_sum']
    prof_val = [epoch, batch, cost_bat]
    prof_attr_type = ['INTEGER', 'INTEGER', 'REAL']
    populate_db(prof_attr, prof_attr_type, prof_val, 
            table_name='profile_cost|ann', usr_time=timestamp)


def profile_net_data(epoch, batch, net, net_ip, timestamp):
    """
    net_ip: raw input data, fed to net
    """
    net_op = net.net_act_forward(net_ip)
    data_attr = ['epoch', 'batch']
    data_attr += ['x{}'.format(i) for i in range(net_ip.shape[1])]
    data_attr += ['y{}'.format(i) for i in range(net_op.shape[1])]
    data_attr_type = ['INTEGER', 'INTEGER'] + ['REAL'] * (net_ip.shape[1] + net_op.shape[1])
    populate_db(data_attr, data_attr_type, array([epoch, batch]), net_ip, net_op, 
            table_name='net_data|ann', usr_time=timestamp)
