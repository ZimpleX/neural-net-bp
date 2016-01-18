"""
sweep script to run net_structure, and collect profile data
"""

import net_structure as ns
import argparse
from conf import *
import re
import sys

import pdb


_STRUCT=['15sig-1lin']
_DATA=['sin_is-3-os-1-ir-(-10~10)-or-(-5~5)']

def parse_args_sweep():
    parser = argparse.ArgumentParser('sweep metrics to obtain profile data')
    parser.add_argument('-r', '--learn_rate', type=float, nargs='+', 
        metavar='LEARN_RATE', help='specify the learning rates that you want to sweep')
    parser.add_argument('-m', '--momentum', type=float, nargs='+', 
        metavar='MOMENTUM', help='specify the momentum that you want to sweep')
    parser.add_argument('-b', '--batch', type=int, nargs='+', 
        metavar='BATCH', help='specify the batch sizes that you want to sweep')
    parser.add_argument('-s', '--struct', type=str, nargs='+', default=_STRUCT,
        metavar='STRUCT', help='specify the net struct you want to explore(excluding input layer)\
        \ne.g.: 15sig-1lin')
    parser.add_argument('-d', '--data', type=str, nargs='+', default=_DATA,
        metavar='DATA', help='specify the data function you want to train')
    return parser.parse_args()


def yield_conf(args):
    r = re.compile("([0-9]+)([a-zA-Z]+)")
    full_struct = [s.split('-') for s in args.struct]
    num_nodes_struct = [[int(r.match(aa).group(1)) for aa in a] for a in full_struct]
    act_nodes_struct = [[r.match(aa).group(2) for aa in a] for a in full_struct]
    args.data = ['{}|ann'.format(d) for d in args.data]
    sys.argv = []
    for r in args.learn_rate:
        for m in args.momentum:
            for b in args.batch:
                for s_idx in range(len(args.struct)):
                    for d in args.data:
                        args_new = ns.parse_args()
                        args_new.batch = b
                        args_new.momentum = m
                        args_new.rate = r
                        args_new.struct = [1] + num_nodes_struct[s_idx]
                        args_new.activation = act_nodes_struct[s_idx]
                        yield args_new


if __name__ == '__main__':
    args = parse_args_sweep()
    for args_new in yield_conf(args):
        ns.net_train_main(args_new)
