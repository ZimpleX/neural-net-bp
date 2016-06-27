"""
"""
import numpy as np
import yaml
from logf.printf import printf
import argparse
from net.structure import net_train_main

import pdb


###########################
#        arg parse        #
###########################
def parse_args():
    parser = argparse.ArgumentParser('neural network model')
    parser.add_argument('yaml_model', type=str, metavar='YAML',
        help='specify the yaml models to be used by this net training')
    parser.add_argument('-p', '--partial_trained', type=str, default=None,
        help='use the partially trained model')
    parser.add_argument('--profile_output', action='store_true',
        help='add this flag if you want to store the net output thoughout epochs')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    from stat_cnn.time import RUNTIME
    yaml_model = yaml.load(open(args.yaml_model))
    tot_time = net_train_main(yaml_model, args)
    for k in RUNTIME.keys():
        printf('{} takes {:.3f}%', k, 100*RUNTIME[k]/tot_time, type='WARN')

