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
        if len(raw[tag].shape) == 1:
            num_entry = raw[tag].shape[0]
            if num_op is None:
                num_op = raw[tag].max() - raw[tag].min() + 1
            z = np.zeros((num_entry, num_op))
            z[np.arange(num_entry), raw[tag]] = 1.
            ret[tag] = z

    np.savez(path_npz, **ret)



def normalize_raw_data(raw, path_out, start_idx, end_idx, num_op=None, scale_axis=(1,2,3)):
    data = raw['data'][start_idx:end_idx]
    target = raw['target'][start_idx:end_idx]
    ret = {}
    # prepare target
    if len(target.shape) == 1:
        num_entry = target.shape[0]
        if num_op is None:
            num_op = target.max() - target.min() + 1
        _ = np.zeros((num_entry, num_op))
        _[np.arange(num_entry), target] = 1.
        ret['target'] = _
    # prepare data
    rmin = np.min(data, axis=scale_axis)
    rmax = np.max(data, axis=scale_axis)
    for a in range(len(scale_axis)):
        rmin = np.expand_dims(rmin, axis=3)
        rmax = np.expand_dims(rmax, axis=3)
    ret['data'] = 2.*(data - rmin)/(rmax-rmin) - 1.

    np.savez(path_out, **ret)



if __name__ == '__main__':
    #output_data = 'train_data/3cat_1000_scale.npz'
    #input_data = 'train_data/3cat_7500.npz'
    #prepare_small_npz(input_data, output_data, 750, 250, 0)
    #normalize_exp_col(output_data, num_op=3)
    path_in = 'train_data/3cat_7500.npz'
    raw = np.load(path_in)
    for s in range(0,7500,1500):
        printf("idx starting: {}", s)
        path_out = 'train_data/raw_bloodcell_part_{}.npz'.format(s)
        normalize_raw_data(raw, path_out, s, s+1500)
