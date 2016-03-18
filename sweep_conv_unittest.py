"""
sweep script for conv unittest on Spark
"""

import argparse
import os
from ec2.EmbedScript import *
from logf.printf import printf
from ec2.cmd import CMD

import pdb


def parse_args():
    parser = argparse.ArgumentParser('sweep to quantatively profile Spark')
    parser.add_argument('--prange', type=int, required=True, nargs=3, help='sweep the data partition scheme: (start, end, stride)')
    parser.add_argument('--irange', type=int, required=True, nargs=3, help='sweep the num of iterations for conv: (start, end, stride)')
    parser.add_argument('--brange', type=int, required=True, nargs=3, help='sweep the num of batches for base mat: (start, end, stride)')
    return parser.parse_args()


def sweep_conv_unittest(args, master_dns):
    count = 0
    for partition in range(*args.prange):
        for itr in range(*args.irange):
            for batch in range(*args.brange):
                sub_args = '-p {} -i {} -b {}'.format(partition, itr, batch)
                cmd = CMD['submit_spark'].format(dns=master_dns, name='neural-net-bp',
                    main='conv_unittest.py', args=sub_args)
                try:    # to be called on EC2 terminal
                    stdout, stderr = runScript(cmd, output_opt='display')
                except ScriptException as se:
                    printf(se, type='ERROR')
                    exit()
                count += 1


def get_master_DNS_from_master():
    cmd = 'curl -s http://169.254.169.254/latest/meta-data/public-hostname'
    try:
        stdout, stderr = runScript(cmd, output_opt='pipe')
        return stdout.decode('utf-8')
    except ScriptException as se:
        printf(se, type='ERROR')
        exit()



if __name__ == '__main__':
    args = parse_args()
    master_dns = get_master_DNS_from_master()
    printf('master DNS:\n{}', master_dns)
    sweep_conv_unittest(args, master_dns)
