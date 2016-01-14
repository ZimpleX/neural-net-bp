"""
processing the data generated by ANN training.
e.g.:
    populate / clear data in db
"""
from numpy import *
from functools import reduce
import pdb
from db_util.basic import *
from db_util.conf import *


"""
The following 4 profile_* functions are wrapper for populate db.
They correspond to 4 tables in the ann.db file
"""

def profile_net_conf(data_dir_name, args, timestamp, db_path=DB_DIR_PARENT):
    """
    INPUT:
        data_dir_name       appended with db_path

    populate the conf of the ANN, timestamp is to identify each run:
    for join of different tables in the future
    """
    struct_info = [str(args.struct[i+1])+args.activation[i] for i in range(len(args.activation))]
    net_struct = '{}lin'.format(args.struct[0])
    net_struct += '-' + reduce(lambda a,b:a+'-'+b, struct_info)

    net_attr = ['struct', 'cost_type', 'train_func', 'test_func', 'train_size', 'test_size',
            'batch_size', 'learn_rate', 'inc_rate', 'dec_rate', 'momentum']
    net_val = array([net_struct, args.cost, args.table_data, args.table_test, args.size_data, args.size_test,
            args.batch, args.rate, args.inc_rate, args.dec_rate, args.momentum])
    net_attr_type = ['TEXT', 'TEXT', 'TEXT', 'TEXT', 'INTEGER', 'INTEGER',
            'INTEGER', 'REAL', 'REAL', 'REAL', 'REAL']
    populate_db(net_attr, net_attr_type, net_val, 
        db_path=db_path+data_dir_name, table_name='meta|ann', 
        usr_time=timestamp, perm='0444', silent=True)


def profile_raw_data_set(data_dir_name, data_set, timestamp, db_path=DB_DIR_PARENT):
    """
    INPUT:
        data_dir_name       appended with db_path

    populate the raw data set that is used to train ANN
    """
    data_attr = ['x{}'.format(i) for i in range(data_set.data.shape[1])]
    data_attr += ['y{}'.format(i) for i in range(data_set.target.shape[1])]
    data_attr_type = ['REAL'] * (data_set.data.shape[1] + data_set.target.shape[1])
    populate_db(data_attr, data_attr_type, data_set.data, data_set.target,
        db_path=db_path+data_dir_name, table_name='raw_data|ann', 
        usr_time=timestamp, perm='0444', silent=True)



def profile_cost(data_dir_name, cost_data, timestamp, db_path=DB_DIR_PARENT):
    """
    INPUT:
        data_dir_name       appended with db_path

    cost_bat is the sum of cost over some batches
    """
    prof_attr = ['epoch_num', 'batch_num', 'cost_sum']
    prof_attr_type = ['INTEGER', 'INTEGER', 'REAL']
    populate_db(prof_attr, prof_attr_type, cost_data,
        db_path=db_path+data_dir_name, table_name='profile_cost|ann', 
        usr_time=timestamp, perm='0444', silent=True)


def profile_net_data(data_dir_name, net_data, net_ip, timestamp, db_path=DB_DIR_PARENT):
    """
    INPUT:
        data_dir_name       appended with db_path

    net_ip: raw input data, fed to net
    """
    data_attr = ['epoch', 'batch']
    data_attr += ['x{}'.format(i) for i in range(net_ip.shape[1])]
    data_attr += ['y{}'.format(i) for i in range(net_data[0][1].shape[1])]
    data_attr_type = ['INTEGER', 'INTEGER'] + ['REAL'] * (net_ip.shape[1] + net_data[0][1].shape[1])
    for dt in net_data:
        populate_db(data_attr, data_attr_type, dt[0], net_ip, dt[1],
            db_path=db_path+data_dir_name, table_name='net_data|ann', 
            usr_time=timestamp, perm='0444', silent=True)
