"""
utility script for design space exploration
"""

import argparse
import os
import yaml
from ec2.EmbedScript import *
from logf.printf import printf
from net.structure import net_train_main
import ec2.conf

import pdb

_MAIN_DIR = '/root/neural-net-bp/'

def parse_args():
    parser = argparse.ArgumentParser('sweep a set of yaml models')
    parser.add_argument('yaml_dir', type=str, help='the directory containing a set of yaml models')
    parser.add_argument('--profile_output', action='store_true',
        help='add this flag if you want to store the net output thoughout epochs')
    parser.add_argument('--slide_method', type=str, choices=['slide_serial', 'slide_spark'],
        default='slide_serial', help='how would you like to do the sliding window? naive serial version or parallelize with spark?')
    parser.add_argument('-p', '--partial_trained', type=str, default=None,
        help='[NOT SUPPORTED NOW]:use the partially trained model')

    return parser.parse_args()


def sweep_training(args):
    # sweep all yamls in dir
    for f in os.listdir(args.yaml_dir):
        yaml_file = '{}/{}'.format(args.yaml_dir, f)
        yaml_model = yaml.load(open(yaml_file))
        net_train_main(yaml_model, args, None)
        ec2_chkpt = yaml_model['checkpoint']
        try:
            s3_dir = args.yaml_dir.strip('/').split('/')[-1]
            s3_file = f.split('.yaml')[0]
            s3_db = '{}.db'.format(s3_file)
            s3_chkpt = '{}.chkpt.npz'.format(s3_file)
            # submit script should already cd to main dir (/root/neural-net-bp)
            cmd = ( "s3_dir={s3_dir}\n"
                    "s3_db={s3_db}\n"
                    "train_name=$(ls -t ./profile_data/ | awk 'NR==1')\n"
                    "aws s3 cp ./profile_data/$train_name/ann.db s3://spark-ec2-log/$train_name/$s3_dir/$s3_db\n"
                    "ec2_chkpt={ec2_chkpt}\n"
                    "s3_chkpt={s3_chkpt}\n"
                    "aws s3 cp $ec2_chkpt s3://spark-ec2-log/$train_name/$s3_dir/$s3_chkpt\n"
                    "\n"
                    "rm $ec2_chkpt\n")\
                .format(s3_dir=s3_dir, s3_db=s3_db, ec2_chkpt=ec2_chkpt, s3_chkpt=s3_chkpt)
            stdout, stderr = runScript(cmd)
        except ScriptException as se:
            printf(se, type='ERROR')






if __name__ == '__main__':
    args = parse_args()
    sweep_training(args)
