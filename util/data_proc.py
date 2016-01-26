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


_PROF_PERM = '0444'

"""
The following 4 profile_* functions are wrapper for populate db.
They correspond to 4 tables in the ann.db file
"""

def profile_net_conf(data_dir_name, yaml_model, timestamp, db_path=DB_DIR_PARENT):
    """
    INPUT:
        data_dir_name       appended with db_path

    populate the conf of the ANN, timestamp is to identify each run:
    for join of different tables in the future
    """
    net_struct = ['{}FC_LIN'.format(yaml_model['input_num_channels'])]
    for l in yaml_model['network']:
        net_struct += ['{}{}'.format(l['num_channels'], l['type'])]
    net_struct = reduce(lambda a,b:a+'-'+b, net_struct)

    net_attr = ['struct', 'cost_type', 'train_func', 'train_size', 'test_size',
            'batch_size', 'learn_rate', 'inc_rate', 'dec_rate', 'momentum']
    net_val = array([net_struct, yaml_model['cost'], yaml_model['obj_name'], yaml_model['data_size'], yaml_model['test_size'],
            yaml_model['batch'], yaml_model['alpha'], yaml_model['inc_rate'], yaml_model['dec_rate'], yaml_model['momentum']])
    net_attr_type = ['TEXT', 'TEXT', 'TEXT', 'INTEGER', 'INTEGER',
            'INTEGER', 'REAL', 'REAL', 'REAL', 'REAL']
    populate_db(net_attr, net_attr_type, net_val, 
        db_path=db_path+data_dir_name, table_name='meta|ann', 
        usr_time=timestamp, perm=_PROF_PERM, silent=True)


def profile_input_data(data_dir_name, timestamp, attr_full, *data_x, db_path=DB_DIR_PARENT):
    """
    INPUT:
        data_dir_name       appended with db_path

    populate the raw data set that is used to train ANN
    """
    data_attr_type = ['REAL'] * len(attr_full)
    attr_full = ['idx'] + attr_full # idx is corresponded to profile_output_data
    data_attr_type = ['INTEGER'] + data_attr_type
    idx_list = [[i+1] for i in range(data_x[0].shape[0])]
    populate_db(attr_full, data_attr_type, idx_list, *data_x,
        db_path=db_path+data_dir_name, table_name='input_data|ann', 
        usr_time=timestamp, perm=_PROF_PERM, silent=True)



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
        usr_time=timestamp, perm=_PROF_PERM, silent=True)



def profile_output_data(data_dir_name, net_data, timestamp, db_path=DB_DIR_PARENT):
    """
    INPUT:
        data_dir_name       appended with db_path

    """
    data_attr = ['idx', 'epoch', 'batch']   # add idx to join with raw_data table
    data_attr += ['y{}'.format(i) for i in range(net_data[0][1].shape[1])]
    data_attr_type = ['INTEGER', 'INTEGER', 'INTEGER'] + ['REAL'] * (net_data[0][1].shape[1])
    idx_list = [[i+1] for i in range(net_data[0][1].shape[0])]
    for dt in net_data:
        populate_db(data_attr, data_attr_type, idx_list, dt[0], dt[1],
            db_path=db_path+data_dir_name, table_name='output_data|ann', 
            usr_time=timestamp, perm=_PROF_PERM, silent=True)
