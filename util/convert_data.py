import numpy as np
import scipy.io
from logf.printf import printf
import logf.filef as ff
import os

from PIL import Image
import copy

#####################################
#  libsvm, mat, npz, h5 conversion  #
#####################################
def libsvm_to_npz(path_libsvm, path_npz, channel, height, width):
    """
    path_libsvm:    input path for libsvm file
    path_npz:       output path for npz file

    data stored in the output npz should be of dimension:
        batch x channel x height x width
    """
    import bob.learn.libsvm as bsvm
    f_libsvm = bsvm.File(path_libsvm)
    label, data = f_libsvm.read_all()
    entries = data.shape[0]
    data = data.reshape(entries, channel, height, width)
    data_compact = {'target': label, 'data': data}
    np.savez(path_npz, **data_compact)


__cat_idx__ = {
    'cha': 0,
    'chg': 1,
    'edu': 2,
    'gym': 3,
    'mer': 4,
    'nav': 5,
    'por': 6,
    'sce': 7,
    'tha': 8,
    'Debris': 0,
    'MCF7': 1,
    'PBMC': 2,
    'THP1': 3
}


def mat_to_npz(path_mat, path_npz, norm_img_key='norm_cell', phase_img_key='phase_cell'):
    """
    path_mat        the parent dir for all categories:
                e.g.,
                    cell/MCF7/1.mat     cell/MCF7/2.mat ...
                    within each *.mat, phase_cell is 2D (e.g., 511 x 511);
                                    norm_cell is 3D (e.g., 511 x 511 x 4)
                    the category name is the sub-dir name, the corresponding index is specified in __cat_idx__
                    Assume that the dir contains *.mat files only
    path_npz        the output dir of converted npz file
                e.g.,
                    the final data will be saved to:
                    path/to/npz/norm.npz        path/to/npz/phase.npz
    """
    ff.mkdir_r(path_npz)
    temp_norm_list = []
    temp_phase_list = []
    count = 0
    for sub in os.listdir(path_mat):
        printf(sub)
        if not os.path.isdir('{}/{}'.format(path_mat, sub)):
            continue
        idx = __cat_idx__[sub]
        mat_dir = '{}/{}'.format(path_mat, sub)
        printf('entering {}', sub)
        npz_norm_data = {'data': None, 'target': None}
        npz_phase_data = {'data': None, 'target': None}
        mat_norm_l = []
        mat_phase_l = []
        idx_l = []
        for m in os.listdir(mat_dir):
            count += 1
            m_path = '{}/{}'.format(mat_dir, m)
            data = scipy.io.loadmat(m_path)
            idx_l += [np.array([[idx]])]
            data_norm = data[norm_img_key]
            data_norm = np.rollaxis(data_norm, 2)[np.newaxis,:]
            data_phase = data[phase_img_key][np.newaxis,np.newaxis,:]
            mat_norm_l += [data_norm]
            mat_phase_l += [data_phase]
        npz_norm_data['data'] = np.concatenate(mat_norm_l, axis=0)
        npz_norm_data['target'] = np.concatenate(idx_l, axis=0)
        npz_phase_data['data'] = np.concatenate(mat_phase_l, axis=0)
        npz_phase_data['target'] = np.concatenate(idx_l, axis=0)
        
        temp_norm = '{}/.temp_{}_norm.npz'.format(path_npz, sub)
        temp_phase = '{}/.temp_{}_phase.npz'.format(path_npz, sub)
        temp_norm_list += [temp_norm]
        temp_phase_list += [temp_phase]
        printf('saving {} norm.npz', sub)
        np.savez(temp_norm, **npz_norm_data)
        printf('saving {} phase.npz', sub)
        np.savez(temp_phase, **npz_phase_data)

    from util.misc_data import npz_concatenate
    npz_concatenate('{}/norm_{}.npz'.format(path_npz, count), *temp_norm_list)
    npz_concatenate('{}/phase_{}.npz'.format(path_npz, count), *temp_phase_list)

    # clear up
    for d in temp_norm_list + temp_phase_list:
        os.remove(d)



#############################
#  array, image conversion  #
#############################
def array_to_img(dataset_in, dir_out, slice_, scale_individual=True, cnn_eval_in=None, start_idx=0):
    """
    dataset_in: dictionary of the format {"data": ndarray, "target": ndarray} (same as training).
    dir_out:    directory storing the output images
    slice_:     the slice for array in dataset_in
    scale_individual
                whether you want to scale the images invidually or altogether.
                the scaling invidually would make the images look 'sharper'.
    cnn_eval_in:
                the classification result from a trained CNN.
    """
    from logf.filef import mkdir_r
    mkdir_r(dir_out)
    try:
        data_in = dataset_in['data'][slice_]
        target_in = dataset_in['target'][slice_]
    except TypeError:
        data_in = dataset_in.data[slice_]
        target_in = dataset_in.target[slice_]
    try:
        data_in = np.asarray(data_in)
        target_in = np.asarray(target_in)
    except MemoryError:
        printf("memory error occurred! Try using smaller slice!", type='ERROR')
        exit()
    entry, channel, height, width = data_in.shape
    data_in = data_in.reshape(entry, channel*height, width)
    target_in = np.nonzero(target_in)[1]
    if cnn_eval_in is None:
        img_out = dir_out + '/img{}_channel{}_category{}.png'
    else:
        img_out = dir_out + '/img{}_channel{}_correct{}_wrong{}.png'
    if scale_individual:
        for i in range(len(data_in)):
            rmin = np.min(data_in[i])
            rmax = np.max(data_in[i])
            data_in[i] = (255/(rmax-rmin)) * (data_in[i] - rmin)
    else:
        rmin = np.min(data_in)
        rmax = np.max(data_in)
        data_in = (255/(rmax-rmin)) * (data_in - rmin)
    for i,img in enumerate(data_in):
        if cnn_eval_in is None:
            Image.fromarray(np.uint8(img))\
                .save(img_out.format(i+start_idx,channel,target_in[i]))
        else:
            Image.fromarray(np.uint8(img))\
                .save(img_out.format(i+start_idx,channel,target_in[i],cnn_eval_in[i].argmax()))

def img_to_array(l_path_img):
    """
    Input a list of images, convert them to ndarray
    """
    _l = [np.asarray(Image.open(img))[np.newaxis,...] \
                for img in l_path_img]
    arr_img = np.concatenate(_l)
    # scale to -1 ~ 1
    arr_img = arr_img/127.5 - 1.
    shape = arr_img.shape
    if len(shape) == 4: # probably RGB image
        return arr_img.transpose((0,3,1,2))
    else:
        assert len(shape) == 3
        # get num of channel
        l_f_img = [path_img.split('/')[-1] for path_img in l_path_img]
        l_chan = [int(f_img.split('_')[1].split('channel')[1])\
                    for f_img in l_f_img]
        assert len(set(l_chan)) == 1
        channel = l_chan[0]
        return arr_img.reshape(shape[0],channel,-1,shape[2])
