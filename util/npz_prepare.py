import numpy as np
from logf.printf import printf

def prepare_small_npz(path_orig, path_out, train_size, valid_size, test_size):
    """
    Assumption: data entries in path_orig have been shuffled already.
    """
    raw = np.load(path_orig)
    ret = {}
    # hard code for now: (target, data) are the keys in path_orig
    ret['train_labels'] = raw['target'][0:train_size]
    ret['validation_labels'] = raw['target'][train_size:(train_size+valid_size)]
    ret['test_labels'] = raw['target'][(train_size+valid_size):(train_size+valid_size+test_size)]
    ret['train'] = raw['data'][0:train_size]
    ret['validation'] = raw['data'][train_size:(train_size+valid_size)]
    ret['test'] = raw['data'][(train_size+valid_size):(train_size+valid_size+test_size)]
    np.savez(path_out, **ret)


def prepare_h5(h5, train_size, valid_size, test_size):
    """

    """
    import tables as tb
    slices = [slice(0,train_size),
        slice(train_size,train_size+valid_size),
        slice(train_size+valid_size,train_size+valid_size+test_size)]
    with tb.openFile(h5, mode='r') as h5f1:
        root1 = h5f1.root
        arr_keys = list(h5f1.get_node(root1)._v_children.keys())
        arr_shape = []
        arr_atom = []
        for k in arr_keys:
            arr_shape += [h5f1.get_node(root1, name=k).shape]
            arr_atom += [h5f1.get_node(root1, name=k).atom]
        h5s = ['{}_{}.h5'.format(h5.split('.h5')[-2],n) for n in ['train','valid','test']]
        batches = [train_size, valid_size, test_size]
        for s,h in enumerate(h5s):
            with tb.openFile(h,mode='w') as f:
                for i,k in enumerate(arr_keys):
                    shape = list(arr_shape[i])
                    shape[0] = batches[s]
                    f.createCArray(f.root,k,arr_atom[i],shape=shape)
                    f.get_node(f.root, name=k)[:] = h5f1.get_node(root1,name=k)[slices[s]]




def partition_small_npz(path_orig, dir_out, small_batch):
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


def normalize_exp_col(path_npz, num_op=None, scale_axis=(1,2,3)):
    raw = np.load(path_npz)
    ret = {}
    for tag in ['train', 'validation', 'test']:
        rmin = np.min(raw[tag], axis=scale_axis)
        rmax = np.max(raw[tag], axis=scale_axis)
        for a in range(len(scale_axis)):
            rmin = np.expand_dims(rmin, axis=3)
            rmax = np.expand_dims(rmax, axis=3)
        ret[tag] = 2.*(raw[tag] - rmin)/(rmax-rmin) - 1.    # normalize

    for tag in ['train_labels', 'validation_labels', 'test_labels']:
        if (len(raw[tag].shape) == 1) or \
            (len(raw[tag].shape) == 2 and 1 in raw[tag].shape):
            num_entry = raw[tag].shape[0]
            raw_tag = raw[tag].flatten().astype(np.int)
            if num_op is None:
                num_op = raw_tag.max() - raw_tag.min() + 1
                tag_min = raw_tag.min()
            else:
                tag_min = 0
            z = np.zeros((num_entry, num_op))
            z[np.arange(num_entry), raw_tag-tag_min] = 1.
            ret[tag] = z

    np.savez(path_npz, **ret)



def normalize_raw_data(raw, path_out, start_idx, end_idx, num_op=None, scale_axis=(1,2,3)):
    """
    raw is loaded npz data;
    path_out is str
    """
    data = raw['data'][start_idx:end_idx]
    target = raw['target'][start_idx:end_idx].astype(int)
    ret = {}
    # prepare target
    if (len(target.shape) == 1) or \
       (len(target.shape) == 2 and 1 in target.shape):
        num_entry = target.shape[0]
        if num_op is None:
            num_op = target.max() - target.min() + 1
            target_min = target.min()
        else:
            target_min = 0
        _ = np.zeros((num_entry, num_op))
        _[np.arange(num_entry), target-target_min] = 1.
        ret['target'] = _
    # prepare data
    rmin = np.min(data, axis=scale_axis)
    rmax = np.max(data, axis=scale_axis)
    for a in range(len(scale_axis)):
        rmin = np.expand_dims(rmin, axis=3)
        rmax = np.expand_dims(rmax, axis=3)
    ret['data'] = 2.*(data - rmin)/(rmax-rmin) - 1.

    np.savez(path_out, **ret)


def split_npz_cat(in_path, out_path_regx, split_key):
    """
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
        



def hdf5_shuffle(hdf5):
    """
    Do the in-place shuffle.
    For efficient memory access, store shuffled array in the new array.
    After the shuffle, the old array can be deleted.
    """
    import timeit
    start_time = timeit.default_timer()
    import tables as tb
    with tb.openFile(hdf5, mode='r+') as h5f:
        root1 = h5f.root
        arr_keys = list(h5f.get_node(root1)._v_children.keys())
        for k in arr_keys:
            arr = h5f.get_node(root1,name=k)
            tot = arr.shape[0]
            for r in range(tot-1,0,-1):     # tot-1, ..., 1
                idx = np.random.randint(0,r+1)
                _temp = arr[r]
                arr[r] = arr[idx]
                arr[idx] = _temp
    end_time = timeit.default_timer()
    printf("time spent on shuffling: {:.3f}", end_time-start_time)
                




if __name__ == '__main__':
    #output_data = 'train_data/3cat_1000_scale.npz'
    #input_data = 'train_data/3cat_7500.npz'
    #prepare_small_npz(input_data, output_data, 750, 250, 0)
    #normalize_exp_col(output_data, num_op=3)

    #path_in = 'train_data/3cat_7500.npz'
    #raw = np.load(path_in)
    #for s in range(0,7500,1500):
    #    printf("idx starting: {}", s)
    #    path_out = 'train_data/raw_bloodcell_part_{}.npz'.format(s)
    #    normalize_raw_data(raw, path_out, s, s+1500)

    import os
    _dir = '../../Temp/temp_norm_npz/'
    f_list = os.listdir(_dir)
    f_list = ['{}/{}'.format(_dir,i) for i in f_list]
    for f in f_list:
        raw = np.load(f)
        normalize_raw_data(raw,f,0,10000, num_op=9)
        printf('done normalizing and expanding {}', f)
