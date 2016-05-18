"""
This script is for converting the libsvm data format to png.

Prerequisite: bob.learn.libsvm python module
"""

from PIL import Image
import numpy as np
import argparse
from logf.printf import printf


def libsvm_to_npz(path_libsvm, path_npz, channel, height, width):
    import bob.learn.libsvm as bsvm
    f_libsvm = bsvm.File(path_libsvm)
    label, data = f_libsvm.read_all()
    entries = data.shape[0]
    data = data.reshape(entries, channel, height, width)
    data_compact = {'target': label, 'data': data}
    np.savez(path_npz, **data_compact)


def npz_concatenate(out_npz, *path_npz):
    """
    concatenating may fail for large datasets, if the ndarray cannot fill into memory.
    --> TODO:
        change to hdf5
    """
    data_l = [np.load(p) for p in path_npz]
    keys = set([tuple(d.keys()) for d in data_l])
    assert len(keys) == 1
    keys = list(keys)[0]
    data_merge = {}
    num_entries = 0
    for k in keys:
        try:
            data_merge[k] = np.concatenate(tuple([dl[k] for dl in data_l]), axis=0)
        except MemoryError:
            printf("out of memory. The data is probably too large.")
            exit()
        if num_entries == 0:
            num_entries = data_merge[k].shape[0]
        else:
            assert num_entries == data_merge[k].shape[0]
    indices = np.arange(num_entries)
    np.random.shuffle(indices)
    for k in keys:
        data_merge[k] = data_merge[k][indices]

    np.savez(out_npz, **data_merge)


def hdf5_concatenate(out_h5, *path_npz):
    """
    concatenate list of npz files into a hdf5.
    This is supposed to be called when the previous npz_concatenate
    function failed (ndarray cannot fit in RAM).
    
    NOTE:
        assume the npz file contains two arrays (data & label), not
        in the format of (train, validation, test).
    """
    # get info from a list of smaller npz files
    tot_size = 0
    shape_list = None
    key_list = None
    dtype_list = None
    for f in path_npz:
        _f = np.load(f)
        _key = _f.keys()
        _cur_shape_list = [_f[ki].shape[1:] for ki in _key]
        _dtype = [_f[ki].dtype for ki in _key]
        if shape_list is None:
            shape_list = _cur_shape_list
            key_list = _key
            dtype_list = _dtype
        else:
            assert shape_list == _cur_shape_list
            assert key_list == _key
            assert dtype_list == _dtype
        tot_size += _f[_key[0]].shape[0]
        printf('getting shape for: {}', f, separator=None)
    shape_list = [(tot_size,)+i for i in shape_list]
    printf("shape of arrays: {}", shape_list)

    # create empty large h5 files
    import tables as tb
    with tb.openFile(out_h5, mode='w', title="huge data") as h5file:
        root = h5file.root
        for i,shape in enumerate(shape_list):
            if dtype_list[i] == np.array(1).dtype:
                _type = tb.Int64Atom()
            elif dtype_list[i] == np.array(1.).dtype:
                _type = tb.Float64Atom()
            else:
                printf("unrecognized data type!", type="ERROR")
                exit()
            h5file.createCArray(root, key_list[i], _type, shape=shape)
        printf("created h5 file: {}", h5file)
        # fill in real data from small files into h5
        cur_idx = 0
        for f in path_npz:
            _f = np.load(f)
            _batch = _f[_f.keys()[0]].shape[0]
            for _k in key_list:
                h5file.get_node(root, name=_k)[cur_idx:cur_idx+_batch] = _f[_k][:]
            cur_idx += _batch
            printf('update data to index: {}', cur_idx)


def parse_args():
    parser = argparse.ArgumentParser('simple data format converter')
    parser.add_argument('path_libsvm', type=str, help='input path of *.libsvm')
    parser.add_argument('path_npz', type=str, help='output path of *.npz')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #libsvm_to_npz(args.path_libsvm, args.path_npz, 1, 256, 256)
    #npz_concatenate('train_data/3cat_7500.npz', 'train_data/3T3_2500.npz', 'train_data/OST_2500.npz', 'train_data/OAC_2500.npz')
