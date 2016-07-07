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
    for img in img_list:
        ip_arr = util.convert_data.img_to_array(img)
        op = net.net_act_forward(ip_arr)
        printf('file: {}',img.split('/')[-1])
        printf('predicted category: {}', op.argmax(axis=1), separator=None)
