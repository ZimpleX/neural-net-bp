import numpy as np
from logf.printf import printf


def h5_prepare(h5, train_size, valid_size, test_size):
    """
    given a h5 file containing all data, prepare 3 small
    files as training, validation and testing data sets.
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



def npz_normalize_reshape_expand(path_in, path_out, dshape=None, start_idx=0, end_idx=None, num_op=None, scale_axis=(1,2,3)):
    """
    path_in is input npz path;
    path_out is output npz path
    """
    raw = np.load(path_in)
    end_idx = (end_idx is None) and len(raw['data']) or end_idx
    data = raw['data'][start_idx:end_idx]
    if dshape is not None:
        data = data.reshape(dshape)
    target = raw['target'][start_idx:end_idx].astype(int)
    ret = {}
    ret['target'] = target
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



