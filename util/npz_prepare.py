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

def normalize_exp_col(path_npz, num_op):
    raw = np.load(path_npz)
    ret = {}
    rmin = raw['train'].min()
    rmax = raw['train'].max()
    ret['train'] = 2.*(raw['train'] - rmin)/(rmax-rmin) - 1.    # normalize
    rmin = raw['validation'].min()
    rmax = raw['validation'].max()
    ret['validation'] = 2.*(raw['validation'] - rmin)/(rmax-rmin) - 1.
    rmin = raw['test'].min()
    rmax = raw['test'].max()
    ret['test'] = 2.*(raw['test'] - rmin)/(rmax-rmin) - 1.

    num_entry = raw['train_labels'].shape[0]
    z = np.zeros((num_entry, num_op))
    z[np.arange(num_entry), raw['train_labels']] = 1.
    ret['train_labels'] = z
    num_entry = raw['validation_labels'].shape[0]
    z = np.zeros((num_entry, num_op))
    z[np.arange(num_entry), raw['validation_labels']] = 1.
    ret['validation_labels'] = z
    num_entry = raw['test_labels'].shape[0]
    z = np.zeros((num_entry, num_op))
    z[np.arange(num_entry), raw['test_labels']] = 1.
    ret['test_labels'] = z
    np.savez(path_npz, **ret)



if __name__ == '__main__':
    # prepare_small_npz('train_data/3cat_7500.npz', 'train_data/3cat_900.npz', 750, 100, 50)
    normalize_exp_col('train_data/3cat_900.npz', 3)
