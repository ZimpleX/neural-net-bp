import numpy as np
from logf.printf import printf


def npz_lossy_normalize(path_in, path_out):
    """
    Normalize each [-1,1] floating point image to [0,255] utf-8,
    then normalize back to [-1,1].
    This is intentional to see how the precision of data types 
    affects the classification results.
    """
    f_in = np.load(path_in)
    d = f_in['data']
    assert d.max() == 1. and d.min() == -1.
    d += 1.
    d /= 2.     # 0~1
    d *= 255.   # 0~255
    d = d.astype(np.uint8).astype(np.float)
    d /= 255    # 0~1
    d -= 0.5
    d *= 2.     # -1~1
    np.savez(path_out, **{'data':d,'target':f_in['target']})


def npz_partition_small(path_orig, dir_out, small_batch):
    """
    partition a single npz with large batch into several small npz with small batch
    """
    raw = np.load(path_orig)
    raw_keys = raw.keys()
    tot_batch = raw[raw_keys[0]].shape[0]
    name_orig = path_orig.split('/')[-1].split('.npz')[0]
    for r in range(0,tot_batch,small_batch):
        cur = {k: raw[k][r:(r+small_batch)] for k in raw_keys}
        f_out = '{}/{}_{}.npz'.format(dir_out, name_orig, r)
        np.savez(f_out, **cur)
        printf('saved {}', f_out)


def npz_split_cat(in_path, out_path_regx, split_key):
    """
    split out several npz files based on the target.
    out_path_regx       e.g., cell_cat{}.npz
    """
    fin = np.load(in_path)
    out_val = set(map(tuple, fin[split_key]))
    for v in out_val:
        v = np.array(v)
        split_out = {}
        idx = (fin[split_key] == v).all(axis=1).nonzero()[0]
        for k in fin.keys():
            split_out[k] = fin[k][idx]
        np.savez(out_path_regx.format(v), **split_out)
 

def npz_concatenate(out_npz, *path_npz):
    """
    concatenating may fail for large datasets, if the ndarray cannot fill into memory.
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

