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
    parser.add_argument('--slide_method', type=str, choices=['slide_serial', 'slide_spark'],
        default='slide_spark', help='how would you like to do the sliding window? naive serial version or parallelize with spark?')
    return parser.parse_args()



if __name__ == '__main__':
    try:
        spark_ok = True
        from pyspark import SparkContext
        sc = SparkContext(appName='ImageNet-dummy')
    except Exception as e:
        spark_ok = False
        sc = None
        printf("No Spark: run SERIAL version of CNN", type="WARN")

    args = parse_args()
    if not spark_ok:
        args.slide_method = 'slide_serial'
    from stat_cnn.time import RUNTIME
    if args.yaml_model:
        yaml_model_full = 'yaml_model/{}.yaml'.format(args.yaml_model)
        yaml_model = yaml.load(open(yaml_model_full))
        tot_time = net_train_main(yaml_model, args, sc)
        for k in RUNTIME.keys():
            printf('{} takes {:.3f}%', k, 100*RUNTIME[k]/tot_time, type='WARN')

