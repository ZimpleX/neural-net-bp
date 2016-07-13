"""
This script is to manually test if a trained model feels good.
"""

from net.structure import Net_structure
import argparse
import numpy as np
import util.convert_data
from logf.printf import printf
import os


def parse_args():
    parser = argparse.ArgumentParser('evaluate the quality of trained model')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file of trained net')
    parser.add_argument('test_img', type=str, help='path to the image to be tested, can be an image or a dir containing images')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    net = Net_structure(None)
    net.import_(args.checkpoint)
    if os.path.isdir(args.test_img):
        img_list = [i for i in os.listdir(args.test_img) if i[0] != '.']
        img_list = ['{}/{}'.format(args.test_img,i) for i in img_list]
    else:
        img_list = [args.test_img]
    ip_arr = util.convert_data.img_to_array(img_list)
    op_arr = net.net_act_forward(ip_arr).argmax(axis=1)

    # print out info
    parent_dir = '/'.join(img_list[0].split('/')[0:-1])
    printf('parent dir: {}', parent_dir,separator=None)
    img_file_list = [f.split('/')[-1] for f in img_list]
    f_max_len = np.array([len(f) for f in img_file_list]).max()
    print('-------------' + '-'*f_max_len)
    print("PREDICTED    INPUT_FILE")
    print('-------------' + '-'*f_max_len)
    for i,f in enumerate(img_file_list):
        print('{:9d}    {}'.format(op_arr[i],f))
