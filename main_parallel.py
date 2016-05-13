"""
Experimental parallelize the training of DCNN, idea borrow from Genetic Algo.

Denote:
    D       total number of training images
    N       number of cores in the cluster
    S       number of images in a smaller pack of training data

Idea:
    *   split D into units of S, distribute them to each core.
    *   each core do training on S individually, with mini-batch (e.g.: 10)
    *   aggregate the N samples of partially trained DCNN, replicate them on N cores again
    *   repeat
"""

import numpy as np
import yaml
from logf.printf import printf
import argparse
import net.structure as ns
from math import ceil

import pdb



def parse_args():
    parser = argparse.ArgumentParser('experimental parallelize training')
    parser.add_argument('-N', '--num_cores', type=int, help='total number of cores in the cluster')
    parser.add_argument('-D', '--data_size', type=int, help='total number of images in the training set')
    parser.add_argument('-S', '--small_data', type=int, help='split D into units of S images')
    return parser.parse_args()


def aggregate_net(net_list):
    """
    aggregate several partially trained nets into one.
    spirit: based on the delta change in weight and bias
    e.g.:
        
    Arguments:
        net_list    the list of updated nets
    """
    for i,w in enumerate(net_list[0].w_list):
        if len(w.shape) == 4:
            # net_list[0].b_list[i] ...    # len of w & b list are the same
            # w ...
        else:   # FC layer
    
    # calculate the measurement / cost of delta
    ##      each feature map has 1 cost / each fc layer has 1 cost

    # reduce based on a tuple (layer/feamap, eval_list)
    


if __name__ == '__main__':
    for itr in ceil(D/(S*N)):
        # replicate net.w_list & net.b_list (partially trained)
        # new_net*N = (S*N).map(lambda: net_train_main)
        ##      net_train_main should return flattened w & b, and the eval calculated based on flattened DELTA w & DELTA b
        ##      then reduce based on:
        ###         i.  w & b, eval_list: simple but with some communication overhead
        ###         ii. eval_list: 
        # (new_net*N).reduce(max delta change for each feature map)


        # reduce on (w_flatterned, core_idx) --> output: [max_eval, ]
        
