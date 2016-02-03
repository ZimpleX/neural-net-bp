import numpy as np


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


def normalize_exp_col(path_npz, num_op, scale_axis=(1,2,3)):
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
        num_entry = raw[tag].shape[0]
        z = np.zeros((num_entry, num_op))
        z[np.arange(num_entry), raw[tag]] = 1.
        ret[tag] = z

    np.savez(path_npz, **ret)



if __name__ == '__main__':
    #prepare_small_npz('train_data/3cat_7500.npz', 'train_data/3cat_900_scale.npz', 750, 100, 50)
    normalize_exp_col('train_data/3cat_900_scale.npz', 3)
