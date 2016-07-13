"""
This script is to manually test if a trained model feels good.
"""

from net.structure import Net_structure
import argparse
import numpy as np
import util.convert_data
from logf.printf import printf
import os
from functools import reduce


def parse_args():
    parser = argparse.ArgumentParser('evaluate the quality of trained model')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file of trained net')
    parser.add_argument('test_file', type=str, help='path to the data/image to be tested, can be a file or a dir')
    return parser.parse_args()


def fname_max_len(f_list):
    """
    get the max string length for files in f_list
    return
        parent_dir, max_len
    """
    parent_dir = '/'.join(f_list[0].split('/')[0:-1]) or '.'
    f_list = [f.split('/')[-1] for f in f_list]
    f_max_len = np.array([len(f) for f in f_list]).max()
    return parent_dir, f_max_len



def img_proc(net, img_list, mini_batch=100):
    op_arr = None
    for i in range(0,len(img_list),mini_batch):
        cur_ip_arr = util.convert_data.img_to_array(img_list[i:i+mini_batch])
        cur_op_arr = net.net_act_forward(cur_ip_arr).argmax(axis=1)
        if op_arr is None:
            op_arr = cur_op_arr
        else:
            op_arr = np.concatenate((op_arr,cur_op_arr))
    # print out info
    parent_dir,f_max_len = fname_max_len(img_list)
    printf('parent dir: {}', parent_dir,separator=None)
    row_info = '{:9}    {:>' + str(f_max_len) + '}'
    print('-------------' + '-'*f_max_len)
    print(row_info.format('PREDICTED','INPUT_FILE'))
    print('-------------' + '-'*f_max_len)
    for i,f in enumerate(img_list):
        print(row_info.format(op_arr[i],f.split('/')[-1]))


def npz_h5_print_info(ret_list,f_list):
    parent_dir, f_max_len = fname_max_len(f_list)
    printf('parent dir: {}', parent_dir,separator=None)
    title_info = '{:>11}    {:>8}    {:>8}    {:>' + str(f_max_len) + '}'
    row_info = '{:11}    {:8.4f}    {:8.4f}    {:>' + str(f_max_len) + '}'
    row_len = 11+8+8+3*4+f_max_len
    print('-'*row_len)
    print(title_info.format('NUM_ENTRIES','ACCURACY','COST','FILE'))
    print('-'*row_len)
    for f_info in ret_list:
        print(row_info.format(f_info[3],f_info[2],f_info[1],f_info[0]))
    print('-'*row_len)
    tot_cost,tot_acc,tot_entry = \
                reduce(lambda x1,x2: (x1[0]+x2[1]*x2[3],
                                      x1[1]+x2[2]*x2[3],
                                      x1[2]+x2[3]),
                                      ret_list, (0.,0.,0))
    print(row_info.format(tot_entry,tot_acc/tot_entry,tot_cost/tot_entry, 'TOTAL'))
 

def npz_proc(net, npz_list):
    ret_list = []   # [(npz,cost,accuracy,tot_img)]
    for npz in npz_list:
        data_tup = np.load(npz)
        data = data_tup['data']
        target = data_tup['target']
        cur_cost,cur_acc = net.evaluate(data,target,eval_details=True,eval_name=npz)
        ret_list += [(npz.split('/')[-1],cur_cost,cur_acc,data.shape[0])]
    npz_h5_print_info(ret_list, npz_list) 


def h5_proc(net, h5_list):
    ret_list = []
    import tables as tb
    for h5 in h5_list:
        data_tup = tb.openFile(h5,mode='r+')
        data = data_tup.root.data
        target = data_tup.root.target
        cur_cost,cur_acc = net.evaluate(data,target,eval_details=True,eval_name=h5,mini_batch=100)
        ret_list += [(h5.split('/')[-1],cur_cost,cur_acc,data.shape[0])]
        data_tup.close()
    npz_h5_print_info(ret_list, h5_list)


test_func = {'png': img_proc,
             'npz': npz_proc,
             'h5':  h5_proc}


if __name__ == '__main__':
    args = parse_args()
    net = Net_structure(None)
    net.import_(args.checkpoint)
    if os.path.isdir(args.test_file):
        file_list = [i for i in os.listdir(args.test_file) if i[0] != '.']
        file_list = ['{}/{}'.format(args.test_file,i) for i in file_list]
    else:
        file_list = [args.test_file]

    file_type = set([f.split('.')[-1] for f in file_list])
    try:
        assert len(file_type) == 1
    except AssertionError:
        printf('the input directory must contain ONLY 1 file type!', type='ERROR')
        exit()
    file_type = list(file_type)[0]

    try:
        assert file_type in test_func.keys()
    except AssertionError:
        printf('the file type {} is not supported!',file_type)
        exit()

    test_func[file_type](net, file_list)
