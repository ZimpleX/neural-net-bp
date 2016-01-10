"""
processing the data generated by ANN training.
e.g.:
    populate / clear data in db
"""
from numpy import *
from functools import reduce
import pdb
from db_util.basic import *


"""
The following 4 profile_* functions are wrapper for populate db.
They correspond to 4 tables in the ann.db file
"""

def profile_net_conf(data_dir_name, args, timestamp):
    """
    INPUT:
        data_dir_name       appended with _DB_DIR_PARENT

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
        db_path=_DB_DIR_PARENT+data_dir_name, table_name='meta|ann', 
        usr_time=timestamp)


def profile_raw_data_set(data_dir_name, data_set, timestamp):
    """
    INPUT:
        data_dir_name       appended with _DB_DIR_PARENT

    populate the raw data set that is used to train ANN
    """
    data_attr = ['x{}'.format(i) for i in range(data_set.data.shape[1])]
    data_attr += ['y{}'.format(i) for i in range(data_set.target.shape[1])]
    data_attr_type = ['REAL'] * (data_set.data.shape[1] + data_set.target.shape[1])
    populate_db(data_attr, data_attr_type, data_set.data, data_set.target,
        db_path=_DB_DIR_PARENT+data_dir_name, table_name='raw_data|ann', 
        usr_time=timestamp)



def profile_cost(data_dir_name, epoch, batch, cost_bat, timestamp):
    """
    INPUT:
        data_dir_name       appended with _DB_DIR_PARENT

    cost_bat is the sum of cost over some batches
    """
    prof_attr = ['epoch_num', 'batch_num', 'cost_sum']
    prof_val = [epoch, batch, cost_bat]
    prof_attr_type = ['INTEGER', 'INTEGER', 'REAL']
    populate_db(prof_attr, prof_attr_type, prof_val,
        db_path=_DB_DIR_PARENT+data_dir_name, table_name='profile_cost|ann', 
        usr_time=timestamp)


def profile_net_data(data_dir_name, epoch, batch, net, net_ip, timestamp):
    """
    INPUT:
        data_dir_name       appended with _DB_DIR_PARENT

    net_ip: raw input data, fed to net
    """
    net_op = net.net_act_forward(net_ip)
    data_attr = ['epoch', 'batch']
    data_attr += ['x{}'.format(i) for i in range(net_ip.shape[1])]
    data_attr += ['y{}'.format(i) for i in range(net_op.shape[1])]
    data_attr_type = ['INTEGER', 'INTEGER'] + ['REAL'] * (net_ip.shape[1] + net_op.shape[1])
    populate_db(data_attr, data_attr_type, array([epoch, batch]), net_ip, net_op,
        db_path=_DB_DIR_PARENT+data_dir_name, table_name='net_data|ann', 
        usr_time=timestamp)
