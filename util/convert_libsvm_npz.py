"""
This script is for converting the libsvm data format to png.

Prerequisite: bob.learn.libsvm python module
"""

from PIL import Image
import bob.learn.libsvm as bsvm
import numpy as np
import argparse


def libsvm_to_npz(path_libsvm, path_npz, channel, height, width):
    f_libsvm = bsvm.File(path_libsvm)
    label, data = f_libsvm.read_all()
    entries = data.shape[0]
    data = data.reshape(entries, channel, height, width)
    data_compact = {'target': label, 'data': data}
    np.savez(path_npz, **data_compact)


def parse_args():
    parser = argparse.ArgumentParser('simple data format converter')
    parser.add_argument('path_libsvm', type=str, help='input path of *.libsvm')
    parser.add_argument('path_npz', type=str, help='output path of *.npz')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    libsvm_to_npz(args.path_libsvm, args.path_npz, 1, 256, 256)
