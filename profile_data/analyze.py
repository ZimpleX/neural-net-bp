"""
analyze data by db operation on raw profiling data
"""

from db_util.interact import *
import argparse

_META_TABLE = 'meta|ann'
_DATA_TABLE = 'profile_cost|ann'

_ATTR_RANGE = [ 'struct',
                'cost_type',
                'train_func',
                'test_func',
                'train_size',
                'test_size',
                'batch_size',
                'learn_rate',
                'inc_rate',
                'dec_rate',
                'momentum']

def parse_args():
    parser = argparse.ArgumentParser('db analysis')
    parser.add_argument('-m', '--meta_table', type=str, metavar='META',
        default=_META_TABLE, help='table containing meta info of training')
    parser.add_argument('-d', '--data_table', type=str, metavar='DATA',
        default=_DATA_TABLE, help='table containing data info of training')
    parser.add_argument('-atr', '--attr_list', type=str, metavar='ATR',
        nargs='+', choices=_ATTR_RANGE, help='want to control what variables?')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    db_control_dim(args.meta_table, args.data_table, *args.attr_list)
