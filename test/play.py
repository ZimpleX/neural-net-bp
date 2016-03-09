"""
This script is to manually test if a trained model feels good.
"""

from net.structure import Net_structure
import argparse
import numpy as np
import util.convert_ndarr_img as img_cvt
from logf.printf import printf


def parse_args():
    parser = argparse.ArgumentParser('evaluate the quality of trained model')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file of trained net')
    parser.add_argument('test_img', type=str, help='path to the image to be tested')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    net = Net_structure(None, None, None)
    net.import_(args.checkpoint)
    ip_arr = img_cvt.img_to_array(args.test_img)
    op = net.net_act_forward(ip_arr)
    printf('predicted category: {}', op.argmax(axis=1))
