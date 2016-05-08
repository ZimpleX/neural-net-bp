from PIL import Image
import numpy as np

from logf.printf import printf
import logf.filef as filef
from net.structure import *

import argparse

import pdb

def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint to visualize')
    parser.add_argument('out_dir', type=str, default='feamap_viz', help='output dir to store the feamap images')
    return parser.parse_args()



def get_w_list(chkpt_f):
    net = Net_structure(None, None, None)
    net.import_(chkpt_f)
    return net.w_list




if __name__ == '__main__':
    args = parse_args()
    filef.mkdir_r(args.out_dir)
    for i,w in enumerate(get_w_list(args.checkpoint)):
        if len(w.shape) != 4:
            continue
        sub_dir = '{}/{}'.format(args.out_dir, 'layer{}'.format(i))
        filef.mkdir_r(sub_dir)
        for j,m in enumerate(w):
            for k,n in enumerate(m):
                img_size = n.shape
                scale = 256//img_size[0]
                cur_min = n.min()
                cur_max = n.max()
                if cur_max == cur_min:
                    n = np.zeros(img_size)
                else:
                    n = 255*(w-cur_min)/(cur_max-cur_min)
                    n = n[0][0]
                n = n.repeat(scale, axis=0)
                n = n.repeat(scale, axis=1)
                Image.fromarray(np.uint8(n)).save('{}/{}_{}.jpg'.format(sub_dir,j,k))
