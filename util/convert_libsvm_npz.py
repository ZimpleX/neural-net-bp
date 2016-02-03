"""
This script is for converting the libsvm data format to png.

Prerequisite: bob.learn.libsvm python module
"""

from PIL import Image
import numpy as np
import argparse


def libsvm_to_npz(path_libsvm, path_npz, channel, height, width):
    import bob.learn.libsvm as bsvm
    f_libsvm = bsvm.File(path_libsvm)
    label, data = f_libsvm.read_all()
    entries = data.shape[0]
    data = data.reshape(entries, channel, height, width)
    data_compact = {'target': label, 'data': data}
    np.savez(path_npz, **data_compact)


def npz_concatenate(out_npz, *path_npz):
    data_l = [np.load(p) for p in path_npz]
    keys = set([tuple(d.keys()) for d in data_l])
    assert len(keys) == 1
    keys = list(keys)[0]
    data_merge = {}
    num_entries = 0
    for k in keys:
        data_merge[k] = np.concatenate(tuple([dl[k] for dl in data_l]), axis=0)
        if num_entries == 0:
            num_entries = data_merge[k].shape[0]
        else:
            assert num_entries == data_merge[k].shape[0]
    indices = np.arange(num_entries)
    np.random.shuffle(indices)
    for k in keys:
        data_merge[k] = data_merge[k][indices]

    np.savez(out_npz, **data_merge)


def parse_args():
    parser = argparse.ArgumentParser('simple data format converter')
    parser.add_argument('path_libsvm', type=str, help='input path of *.libsvm')
    parser.add_argument('path_npz', type=str, help='output path of *.npz')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #libsvm_to_npz(args.path_libsvm, args.path_npz, 1, 256, 256)
    #npz_concatenate('train_data/3cat_7500.npz', 'train_data/3T3_2500.npz', 'train_data/OST_2500.npz', 'train_data/OAC_2500.npz')
