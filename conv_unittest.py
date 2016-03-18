"""
Unittest of purely convolution operation:
For example: 
    convolve [256 x 256] with [11 x 11]
"""

import numpy as np
from logf.printf import printf
import argparse
import timeit
from functools import reduce

import pdb

np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser('unit test for convolution profiling')
    parser.add_argument('--base_mat', type=int, nargs=4, default=[400,32,128,128], help='[batch, chan_0, row_n, col_n]')
    parser.add_argument('--kern_mat', type=int, nargs=4, default=[32,32,11,11], help='[chan_n1, chan_0, kern, kern]')
    parser.add_argument('-p', '--partition', type=int, default=8, help='num of partitions for base mat')
    parser.add_argument('-i', '--itr', type=int, default=1, help='number of times the kern mat is convolved with base mat')
    parser.add_argument('-b', '--batch', type=int, default=None, help='specifically set batch size (overwrite args.base_mat[0])')
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
    

def get_app_name(args):
    template = 'conv_unit-CPUload_{}-partition_{}-itr_{}'
    CPUload = reduce(lambda _1,_2: _1*_2, args.base_mat+args.kern_mat)
    CPUload /= args.kern_mat[1]
    CPUload = CPUload // 1000000    # in terms of Million
    partition = args.partition
    itr = args.itr
    return template.format(CPUload, partition, itr)


def prepare_conf(base_mat, kern_mat, sliding_stride):
    assert kern_mat.shape[0] == kern_mat.shape[1]
    assert kern_mat.shape[2] == kern_mat.shape[3]
    assert base_mat.shape[2] == base_mat.shape[3]
    # to ensure that output mat x,y dim is the same as base_mat x,y
    dim = base_mat.shape[2]
    kern_dim = kern_mat.shape[2]
    padding2 = (dim-1)*sliding_stride + kern_dim - dim
    assert padding2 % 2 == 0
    import conv.slide_win
    f = conv.slide_win.slid_win_4d_flip    # lambda for rdd.map
    f_conv = conv.slide_win.convolution()
    return {'patch_stride': 1,
            'sliding_stride': sliding_stride,
            'padding': padding2/2,
            'f': f,
            'f_conv': f_conv}

       

def do(base_mat, kern_mat, ss, ps, padding, f, f_conv, partition, itr, sc):
    tot_batch = base_mat.shape[0]
    # assert tot_batch%partition == 0
    from math import ceil
    unit_batch = int(ceil(tot_batch / partition))
    l_base = []
    for p in range(0,tot_batch,unit_batch):
        l_base += [base_mat[p:(p+unit_batch)]]

    _rdd = sc.parallelize(l_base, partition)
    for i in range(itr):
        _rdd = _rdd.map(lambda _: f(_,kern_mat,ss,ps,padding,f_conv,None))
    _ret = _rdd.collect()
    _ret = reduce(lambda _1,_2: np.concatenate((_1,_2),axis=0), _ret)
    assert _ret.shape == base_mat.shape
    return _ret



if __name__ == '__main__':
    args = parse_args()
    if args.batch is not None:
        args.base_mat[0] = args.batch
    data = data_gen(args)
    try:
        from pyspark import SparkContext
        sc = SparkContext(appName=get_app_name(args))
    except Exception as e:
        printf('Failed to start Spark!', type='ERROR')
        exit()
    c = prepare_conf(data['input'], data['kern'], 1)
    ret = do(data['input'], data['kern'], c['sliding_stride'], c['patch_stride'],
        c['padding'], c['f'], c['f_conv'], args.partition, args.itr, sc)
    printf('sum of all elements:\n{}',ret.sum())
    printf('ret max shape: {}', ret.shape)
    from stat_cnn.time import RUNTIME
    for k in RUNTIME.keys():
        if RUNTIME[k] != 0.:
            printf('{} takes {:.3f}s', k, RUNTIME[k], type='WARN')
