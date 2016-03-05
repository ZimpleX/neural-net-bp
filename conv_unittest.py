"""
Unittest of purely convolution operation:
For example: 
    convolve [256 x 256] with [11 x 11]
"""

import numpy as np
from logf.printf import printf
import argparse

import pdb


def parse_args():
    parser = argparse.ArgumentParser('unit test for convolution profiling')
    parser.add_argument('--base_mat', type=int, nargs=4, default=[100,3,256,256], help='[batch, chan_0, row_n, col_n]')
    parser.add_argument('--kern_mat', type=int, nargs=4, default=[5,3,11,11], help='[chan_n1, chan_0, kern, kern]')
    return parser.parse_args()


def data_gen(args):
    """
    we are only doing profiling, so the value of data does not matter.
    """
    from conv.util import ff_next_img_size
    out_y, out_x = ff_next_img_size(args.base_mat[2:], args.kern_mat[-1], args.kern_mat[-1]-1, 1)
    shape = (args.base_mat[0], args.kern_mat[0]) + (out_y, out_x)
    return {'input': np.random.randn(*(args.base_mat)),
            'output': np.random.randn(*shape),
            'kern': np.random.randn(*(args.kern_mat))}
    

if __name__ == '__main__':
    args = parse_args()
    assert args.kern_mat[-1] == args.kern_mat[-2]
    data = data_gen(args)
    try:
        spark_ok = True
        from pyspark import SparkContext
        sc = SparkContext(appName='conv-dummy')
    except Exception as e:
        spark_ok = False
        sc = None
        printf('No Spark: run SERIAL version of conv', type='WARN')

    if spark_ok:
        from conv.slide_win_spark import slid_win_4d_flip
        from conv.slide_win_spark import convolution
    else:
        from conv.slide_win import slid_win_4d_flip
        from conv.slide_win import convolution
 
    slid_win_4d_flip(data['input'], data['kern'], 1, 1, args.kern_mat[-1]-1, convolution(), sc)
    from stat_cnn.time import RUNTIME
    for k in RUNTIME.keys():
        if RUNTIME[k] != 0.:
            printf('{} takes {:.3f}s', k, RUNTIME[k], type='WARN')
