"""
This script is to evaluate a batch of images
"""

from net.structure import Net_structure
import argparse
import numpy as np
from logf.printf import printf
from functools import reduce
import io
from net.conf import *
from ec2.EmbedScript import *


def parse_args():
    parser = argparse.ArgumentParser('evaluate the trained model on a batch of test images')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint file of trained net')
    parser.add_argument('test_batch_dir', type=str, help='dir of the batch dataset to be tested')
    parser.add_argument('--partition', required=True, type=int, help='number of partitions for input data\n[NOTE]: you should enforce this when with another flag when launching spark\n[E.G.]: --total-executor-cores')
    parser.add_argument('--num_sets', required=True, type=int, help='number of small sets for evaluation')
    return parser.parse_args()

# ref: http://stackoverflow.com/questions/26884871/advantage-of-broadcast-variables
def evaluate_static(batch_ipt, target, net_bc, mini_batch=0):
    # net = net_bc.value
    num_entry = batch_ipt.shape[0]
    cur_cost = 0.
    mini_batch = (mini_batch <= 0) and num_entry or mini_batch
    cur_correct = 0.
    for k in range(0, num_entry, mini_batch):
        cur_batch = batch_ipt[k:(k+mini_batch)]
        cur_target = target[k:(k+mini_batch)]
        layer_out = cur_batch
        for i,a in enumerate(net_bc.value.activ_list):
            layer_out = a.act_forward(layer_out, net_bc.value.w_list[i], net_bc.value.b_list[i], sc=None)
        cur_cost += sum(net_bc.value.cost.act_forward(layer_out,cur_target))
        if net_bc.value.cost == cost_dict['CE']:
            cur_correct += (cur_target.argmax(axis=1)==layer_out.argmax(axis=1)).sum()
    return cur_cost/num_entry, cur_correct/num_entry


temp_dir = '/_temp'
hdfs_bin = '/root/ephemeral-hdfs/bin/'
def restore_hdfs_data(eval_dir):
    cmd = 'if {hdfs_bin}/hadoop fs -test -d {temp_dir}; then {hdfs_bin}/hadoop fs -mv {temp_dir}/* {eval_dir}; fi'.format(hdfs_bin=hdfs_bin, temp_dir=temp_dir, eval_dir=eval_dir)
    try:
        stdout, stderr = runScript(cmd)
    except ScriptException as se:
        printf(se, type='ERROR')


def setup_hdfs_data(num_sets, eval_dir):
    restore_hdfs_data(eval_dir)
    cmd='{hdfs_bin}/hadoop fs -ls {eval_dir} | grep {eval_dir}'.format(hdfs_bin=hdfs_bin, eval_dir=eval_dir)
    try:
        stdout, stderr = runScript(cmd)
        inf_l = stdout.decode('utf-8').strip().split('\n')
        f_l = [i.split()[-1] for i in inf_l]
        tot_sets = len(f_l)
    except ScriptException as se:
        printf(se, type='ERROR')
    cmd = 'if ! {hdfs_bin}/hadoop fs -test -d {temp_dir}; then {hdfs_bin}/hadoop fs -mkdir {temp_dir}; fi'.format(hdfs_bin=hdfs_bin, temp_dir=temp_dir)
    try:
        stdout, stderr = runScript(cmd)
    except ScriptException as se:
        printf(se, type='ERROR')
    try:
        for d in f_l[0:(tot_sets-num_sets)]:
            stdout, stderr = runScript('{hdfs_bin}/hadoop fs -mv {data} {temp_dir}'.format(hdfs_bin=hdfs_bin, data=d, temp_dir=temp_dir))
    except ScriptException as se:
        printf(se, type='ERROR')




if __name__ == '__main__':
    args = parse_args()
    d_size = args.num_sets * 250
    try:
        from pyspark import SparkContext
        sc = SparkContext(appName='batch_eval-dsize_{}-partition_{}-itr_{}'.format(d_size,args.partition,1))
    except Exception as e:
        sc = None
        printf('No Spark: run SERIAL version of CNN', type='WARN')
    ####################
    #  Serial version  #
    ####################
    if sc is None:
        import os
        net = Net_structure(None, None, None)
        net.import_(args.checkpoint)
        tot_batch = 0
        tot_cost = 0.
        tot_correct = 0.
        for f in os.listdir(args.test_batch_dir):
            cur_f = '{}/{}'.format(args.test_batch_dir, f)
            test = np.load(cur_f)
            cur_batch = test['data'].shape[0]
            cur_cost, cur_correct = net.evaluate(test['data'], test['target'], mini_batch=150, eval_details=True, eval_name=f)
            tot_batch += cur_batch
            tot_cost += cur_cost * cur_batch
            tot_correct += cur_correct * cur_batch
            printf('done testing {}\ncost: {:.3f}, correct: {:.3f}%', cur_f, cur_cost, 100*cur_correct)
        cost = tot_cost / tot_batch
        correct = tot_correct / tot_batch
        printf('done testing {} images\ncost: {:.3f}, correct: {:.3f}%', tot_batch, cost, 100*correct)
        exit()
    
    #####################
    #  Cluster version  #
    #####################
    # setup_hdfs_data(args.num_sets, args.test_batch_dir)
    slide_method = 'slide_serial'

    out_str = ( "Done testing on {} images\n"
                "average cost: {:.3f}\n"
                "average accuracy: {:.3f}%")
    num_img = 0
    num_set = 0
    avg_cost_tot = 0.
    avg_accuracy_tot = 0.
    # distributed load
    mini_batch = 0
    hdfs_files = '{}/*.npz'.format(args.test_batch_dir)
    f_set = sc.binaryFiles(hdfs_files)
    
    # broadcast net
    # dist_data.zipWithIndex().map(lambda _: (_[1],_[0]))
    # Net_structure(None,None,None).import_(args.checkpoint, slide_method=slide_method, sc=None)

    model = io.BytesIO(open(args.checkpoint,'rb').read())
    net = Net_structure(None,None,None)
    net.import_(model, slide_method=slide_method, sc=None)
    net_bc = sc.broadcast(net)
    printf('finish broadcasting net')

    dist_data = f_set.map(lambda _: ( _[0],dict(np.load(io.BytesIO(_[1]))) ))   #.zipWithIndex().map(lambda _: (_[1],_[0]))
    # NOTE: cogroup / join will invoke a shuffle --> too expensive
    # NOTE: http://www.sparktutorials.net/spark-broadcast-variables---what-are-they-and-how-do-i-use-them
    # dist_net_data = dist_net.cogroup(dist_data)
    # exit()
    # (zipIndex, (net, (file_name, data_array)))
    # result_rdd = dist_net_data.map(lambda _: ( _[1][1][0], _[1][0].evaluate(_[1][1][1]['data'],_[1][1][1]['target'],mini_batch=mini_batch) )).collect()
    #________________
    # result_rdd = dist_data.mapPartitions(lambda _: evaluate_static(_[1]['data'],_[1]['target'],net_bc,mini_batch=mini_batch) ,preservesPartitioning=True)
    result_rdd = dist_data.map(lambda _: (_[0], \
        evaluate_static(_[1]['data'],_[1]['target'],net_bc,mini_batch=mini_batch)) )
    result_rdd = result_rdd.collect()

    num_img = dist_data.map(lambda _: _[1]['data'].shape[0]).reduce(lambda _1,_2: _1+_2)
    printf('result:\n{}',result_rdd)
    
    num_set = len(result_rdd)
    avg_cost_tot = reduce(lambda _1,_2: _1+_2[1][0], result_rdd, 0)
    avg_accuracy_tot = reduce(lambda _1,_2: _1+_2[1][1], result_rdd, 0)

       
    avg_cost_tot /= num_set
    avg_accuracy_tot /= num_set
    printf(out_str, num_img, avg_cost_tot, 100*avg_accuracy_tot)

    # restore_hdfs_data(args.test_batch_dir)
