"""
Generate training data for the neural net, store them in the sqlite3 db
"""
from util.training_data_func import trainingFunc
import sqlite3
from logf.printf import *
import db_util as db
import conf
from conf import activation_dict
from cost import cost_dict
import argparse
from random import uniform
import os

import pdb


_NUM_NODES_RANGE = {'input': range(1,11),
                    'output': range(1,11)}
_NUM_NODES_DEFAULT = {'input': 3,
                    'output': 1}
_VAL_RANGE = {'input': [-1,1],
            'output': [-5,5]}
_DATA_SIZE_DEFAULT = 12
_DATA_SIZE_RANGE = range(3, 15)

_FUNC_DEFAULT = 'sin'

func_choices = ['sigmoid',
                'lin',
                'sin',
                'random',
                'ann']
_DB_PATH = conf.TRAINING_DIR

def parseArg():
    parser = argparse.ArgumentParser("generating training data for ANN")
    parser.add_argument('-is', '--input_size', type=int, metavar='M',
            choices=_NUM_NODES_RANGE['input'], default=_NUM_NODES_DEFAULT['input'],
            help='specify the num of inputs the data have')
    parser.add_argument('-os', '--output_size', type=int, metavar='S',
            choices=_NUM_NODES_RANGE['output'], default=_NUM_NODES_DEFAULT['output'],
            help='specify the num of outputs the data have')
    parser.add_argument('-ir', '--input_range', type=int, metavar='IPR', 
            nargs=2, default=_VAL_RANGE['input'], help='range of input: min, max')
    parser.add_argument('-or', '--output_range', type=int, metavar='OPR',
            nargs=2, default=_VAL_RANGE['output'], help='range of output: min, max')
    parser.add_argument('-ds', '--data_size', type=int, metavar='EXP',
            choices=_DATA_SIZE_RANGE, default=_DATA_SIZE_DEFAULT, 
            help='how many entries you need? (in terms of 2^EXP)')
    parser.add_argument('-f', '--function', type=str, 
            choices=func_choices, default=_FUNC_DEFAULT,
            help='specify the func to generate data')
    parser.add_argument('-n', '--db_name', type=str, metavar='DB',
            default=conf.DB_DATA, help='provide the name of database (include file extension)')
    parser.add_argument('-p', '--db_path', type=str, metavar='PATH',
            default=_DB_PATH, help='provide the path of database')

    # following is only for training func of ann
    parser.add_argument("--struct", type=int, metavar='NET_STRUCT', nargs='+',
            default=conf.STRUCT, help='[ANN-bp]: specify the structure of the ANN (num of nodes in each layer)')
    parser.add_argument('--activation', type=str, metavar='NET_ACTIVATION',
            default=conf.ACTIVATION, nargs='+', help='[ANN-bp]: specify the activation of each layer',
            choices=activation_dict.keys())
    parser.add_argument('--cost', type=str, metavar='COST', 
            default=conf.COST, help='[ANN-bp]: specify the cost function',
            choices=cost_dict.keys())

    return parser.parse_args()


def dataGeneratorMain(args):
    db_fullpath = '{}/{}'.format(args.db_path, args.db_name)
    func = args.function
    table = '{}_is-{}-os-{}-ir-({}~{})-or-({}~{})|ann'\
        .format(func, args.input_size, args.output_size, 
                args.input_range[0], args.input_range[1],
                args.output_range[0], args.output_range[1])
    if func == 'ann':
        table = '{}_struct-{}-act-{}-cost-{}|ann'\
            .format(func, args.struct, args.activation, args.cost)
        args.input_size = args.struct[0]
        args.output_size = args.struct[-1]
    new_entry = pow(2, args.data_size)
    if os.path.exists(db_fullpath):
        # set new_entry: how many more entries are needed?
        if db.util.is_table_exist(db_fullpath, table):
            orig_entry = db.util.count_entry(db_fullpath, table)
            new_entry = (new_entry>orig_entry) and (new_entry-orig_entry) or 0
    # populate data into db
    genY, attr_list = trainingFunc(args.function, args.input_size, args.output_size)
    type_list = ['REAL'] * len(attr_list)
    dataList = None
    for i in range(0, new_entry):
        xList = [uniform(args.input_range[0], args.input_range[1]) for k in range(0, args.input_size)]
        if args.function == 'ann':
            xyList = genY(xList, args.struct, args.activation, args.cost)
        else:
            xyList = genY(xList, args.output_range)
        if dataList is None:
            dataList = [xyList]
        else:
            dataList += [xyList]

    if dataList:
        db.basic.populate_db(attr_list, type_list, dataList, db_path=args.db_path, db_name=args.db_name, table_name=table)
        
                


if __name__ == '__main__':
    args = parseArg()
    dataGeneratorMain(args)
