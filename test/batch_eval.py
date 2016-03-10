"""
This script is to evaluate a batch of images
"""

from net.structure import Net_structure
import argparse
import numpy as np
from logf.printf import printf
import os


def parse_args():
    parser = argparse.ArgumentParser('evaluate the trained model on a batch of test images')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file of trained net')
    parser.add_argument('test_batch_dir', type=str, help='dir of the batch dataset to be tested')
    return parser.parse_args()




if __name__ == '__main__':
    args = parse_args()
    net = Net_structure(None, None, None)
    net.import_(args.checkpoint)
    out_str = ( "Done testing on {} images\n"
                "average cost: {:.3f}\n"
                "average accuracy: {:.3f}%")
    num_img = 0
    num_set = 0
    avg_cost_tot = 0.
    avg_accuracy_tot = 0.
    for f in os.listdir(args.test_batch_dir):
        num_set += 1
        printf('f: {}', f, separator=None)
        dataset = np.load('{}/{}'.format(args.test_batch_dir, f))
        num_img += dataset['data'].shape[0]
        avg_cost, avg_accuracy = net.evaluate(dataset['data'], dataset['target'], mini_batch=50)
        avg_cost_tot += avg_cost
        avg_accuracy_tot += avg_accuracy
        printf(out_str, dataset['data'].shape[0], avg_cost, 100*avg_accuracy)
    
    avg_cost_tot /= num_set
    avg_accuracy_tot /= num_set
    printf(out_str, num_img, avg_cost_tot, 100*avg_accuracy_tot)
