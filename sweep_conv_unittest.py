"""
sweep script for conv unittest on Spark
"""

import argparse
import os
from ec2.EmbedScript import *
from logf.printf import printf

import pdb


def parse_args():
    parser = argparse.ArgumentParser('sweep to quantatively profile Spark')
    parser.add_argument('--sub_args', type=str, required=True, help='args to be passed to the spark_submit script')
    parser.add_argument('--prange', type=int, required=True, nargs=3, help='sweep the data partition scheme: (start, end, stride)')
    parser.add_argument('--irange', type=int, required=True, nargs=3, help='sweep the num of iterations for conv: (start, end, stride)')
    parser.add_argument('--brange', type=int, required=True, nargs=3, help='sweep the num of batches for base mat: (start, end, stride)')
    return parser.parse_args()


def sweep_conv_unittest(args):
    count = 0
    for partition in range(*args.prange):
        for itr in range(*args.irange):
            for batch in range(*args.brange):
                if count == 0:
                    sub_args = args.sub_args
                else:
                    sub_args = ' '.join(sub_args.split('--clone'))
                    sub_args = ' '.join(sub_args.split('--scp'))
                    sub_args = ' '.join(sub_args.split('--hdfs'))
                args_main = '\' -p {} -i {} -b {}\''.format(partition,itr,batch)
                cmd = "python3 -m ec2.spark_submit {sub_args} --main {main} --args_main {args_main}"\
                    .format(sub_args=sub_args,
                            main='conv_unittest.py',
                            args_main=args_main)
                try:
                    stdout, stderr = runScript(cmd, output_opt='display')
                except ScriptException as se:
                    printf(se, type='ERROR')
                    exit()
                count += 1



if __name__ == '__main__':
    args = parse_args()
    sweep_conv_unittest(args)
