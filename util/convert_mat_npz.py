import scipy.io
import numpy as np
from logf.printf import printf
import logf.filef as ff
import os
from PIL import Image
import os

import pdb

__cat_idx__ = {
    'cha': '0',
    'chg': '1',
    'edu': '2',
    'gym': '3',
    'mer': '4',
    'nav': '5',
    'por': '6',
    'sce': '7',
    'tha': '8',
    'Debris': '0',
    'MCF7': '1',
    'PBMC': '2',
    'THP1': '3'
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

    from util.convert_libsvm_npz import npz_concatenate
    try:
        npz_concatenate('{}/norm_{}.npz'.format(path_npz, count), *temp_norm_list)
        npz_concatenate('{}/phase_{}.npz'.format(path_npz, count), *temp_phase_list)
    except Exception as e:
        printf('error concatenating (probably out of memory): {}', e)
        printf('keeping temp files: {}', temp_norm_list+temp_phase_list)
        exit()

    # clear up
    for d in temp_norm_list + temp_phase_list:
        os.remove(d)
        
