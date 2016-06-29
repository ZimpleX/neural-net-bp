"""
Analyze data by db operation on raw profiling data
This script is tailor-made for the ANN application, built on top of db_util.interact module
"""

from db_util.interact import *
from db_util.util import *
import argparse
from logf.printf import *
from db_util.conf import *

_META_TABLE = 'meta|ann'
_DATA_TABLE = 'profile_cost|ann'
_DATA_TABLE_CHOICES = [ 'profile_cost|ann',
                        'output_data|ann']


_ATTR_RANGE = [ 'struct',
                'cost_type',
                'train_func',
                'test_func',
                'train_size',
                'test_size',
                'batch_size',
                'learn_rate',
                'momentum']

def parse_args():
    parser = argparse.ArgumentParser('db analysis, specific for ANN application')
    parser.add_argument('-t', '--data_table', type=str, metavar='DATA',
        default=_DATA_TABLE, choices=_DATA_TABLE_CHOICES, help='table containing data info of training')
    parser.add_argument('-atr', '--attr_list', type=str, metavar='ATR',
        nargs='+', choices=_ATTR_RANGE, help='want to control what variables?')
    parser.add_argument('-d', '--db_name', type=str, metavar='DB_NAME',
        default='ann.db', help='provide the name of db file. e.g.: ann.b')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.data_table == 'profile_cost|ann':
        temp_table = ANALYSIS_TABLE_COST
    elif args.data_table == 'output_data|ann':
        temp_table = ANALYSIS_TABLE_OUTPUT
    else:
        temp_table = ANALYSIS_TABLE
        printf('table is neither cost nor output. DOUBLE CHECK!', type='WARN')
    db_control_dim(_META_TABLE, args.data_table, *args.attr_list, 
            temp_table=temp_table, db_name=args.db_name)
    if args.data_table == 'output_data|ann':
        join_input_output_table(ANALYSIS_DB, DB_NAME)
