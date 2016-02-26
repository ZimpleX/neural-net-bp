"""
define the general structure of the whole network, including:
    num of nodes in each layer;
    activate function of each layer;
    deritive of the active function (w.r.t weight / output of prev layer / bias)


TODO:
    * could try generating data by the neural net itself, then see if the net can learn it
    * how to escape local minimum
    * sum Cost function w.r.t all examples in one mini-batch
      when obtaining delta, divide by size of mini-batch
    * adapt learning rate
    * add gradient checking using finite difference
"""

import numpy as np
import yaml
from node_activity import *
import conv.slide_win as slide
from cost import *
from data_setup import *
from conf import *
import util.data_proc as data_util
from logf.stringf import *
from logf.printf import *
from logf.filef import *
import argparse
from time import strftime
import timeit
import sys
import copy

import pdb

np.random.seed(0)

_LOG_FILE = {'net': '{}-net',
            'conf': '{}-conf'}

# profiling the time spent on each component
_TIME = {'checkpoint': 0.,
        'ff': 0.,
        'bp': 0.}

# TODO: make a data class, store raw data (current mini batch)
# and output of each layer
class Net_structure:
    """
    glossary:
    ---------
             layer:     num layer, excludes the input layer
                        layer 0, input layer
                        layer 1, first hidden layer
                        ...
                        layer m, output layer
                 f:     function of activation (e.g.: sigmoid)
                 y:     output of a neuron (node), index rule:
                        y_ij is the jth node on the ith layer
                 w:     weight between 2 adjacent layers, index rule:
                        w_abc is the weight between layer a and (a+1),
                        connecting bth node on layer a and cth node 
                        on layer (a+1)
                 b:     bias between 2 adjacent layers, index rule:
                        b_ij is the bias on the jth node on layer i
                 c:     cost: e.g., sum[(y - target)^2]
          c_d_y[i]:     partial derivative of cost w.r.t. ith layer y
    c_d_w[a][b][c]:     partial derivative of cost w.r.t. ath layer w
                        connecting bth & cth node on the 2 layers
       c_d_b[i][j]:     partial derivative of cost w.r.t. ith layer b
                        operating on the jth node on that layer
    """
    def __init__(self, yaml_model):
        """
        layer_size_list     let the num of nodes in each layer be Ni --> [N1,N2,...Nm]
                            including the input and output layer
        activ_list          list of node activation object, defining the 
                            activation function as well as derivatives
                            length of list = num of layers excluding input layer
        cost_type           the class representing the chosen cost function
        """
        if yaml_model is None:
            return
        self.num_layer = len(yaml_model['network'])    # yaml don't include input layer
        self.w_list = [None] * self.num_layer
        self.dw_list = [None] * self.num_layer
        self.b_list = [None] * self.num_layer
        self.db_list = [None] * self.num_layer
        self.activ_list = [None] * self.num_layer
        self.cost = cost_dict[yaml_model['cost']]
        idx = 0
        prev_chn = yaml_model['input_num_channels']
        prev_img = None
        for l in yaml_model['network']:
            init_wt = ('init_wt' in l) and l['init_wt'] or 0.0
            cur_chn = l['num_channels']
            act_func = activation_dict[l['type']]
            if l['type'] == 'CONVOLUTION' or l['type'] == 'MAXPOOL':
                if prev_img is None:
                    prev_img = (yaml_model['input_image_size_y'], yaml_model['input_image_size_x'])
                kern = l['kernel_size']
                w_shape = (prev_chn, cur_chn, kern, kern)
                act_init = (l['type']=='MAXPOOL') and [kern] or []
                act_init += [l['stride'], l['padding']]
                self.activ_list[idx] = act_func(*act_init)
                prev_img = slide.ff_next_img_size(prev_img, kern, l['padding'], l['stride'])
            else:
                if prev_img is not None:
                    prev_chn = prev_chn * prev_img[0] * prev_img[1]
                    prev_img = None
                w_shape = (prev_chn, cur_chn)
                self.activ_list[idx] = act_func
            self.w_list[idx] = init_wt * np.random.randn(*w_shape)
            self.b_list[idx] = np.zeros(cur_chn)
            self.dw_list[idx] = np.zeros(w_shape)
            self.db_list[idx] = np.zeros(cur_chn)
            prev_chn = cur_chn
            idx += 1

        # store the output of each layer
        self.y_list = [None] * (self.num_layer + 1)
        # current epoch / batch: for debugging & logging
        self.epoch = 0
        self.batch = 0


    def set_w_b(self, w_list, b_list):
        """
        setter for weight and bias matrix
        """
        self.w_list = w_list
        self.b_list = b_list

    def __str__(self):
        """
        print the value of weight and bias array, for each layer
        """
        net_stat = stringf('EPOCH {}', self.epoch, type='NET')
        for i in range(self.num_layer-1, -1, -1):
            num_nodes = self.w_list[i].shape[1]
            layer_str = 'layer {}: {} nodes'
            net_stat = ('{}\n'
                        '{}\n'
                        'weight\n'
                        '{}\n'
                        'bias\n'
                        '{}\n'
                        'd_weight\n'
                        '{}\n'
                        'd_bias\n'
                        '{}').format(net_stat, 
                            stringf(layer_str, i+1, num_nodes, type=None, separator='-'),
                            self.w_list[i], self.b_list[i], self.dw_list[i], self.db_list[i])
        return '{}\n{}\n'.format(net_stat, stringf(layer_str, 0, self.w_list[0].shape[0], type=None, separator='-'))

    def net_act_forward(self, data):
        """
        action: 
            data = activity((data dot weight) + bias)
        argument:
            data    matrix of input data, could be high dimension,
                    but the last 2 dimensions should be like this:
                    n-1 = columns: num of data pieces (batch size)
                    n = rows: num of input nodes
        return:
            layer_out   the output from the output layer
        """
        self.y_list[0] = data
        layer_out = data
        for i in range(self.num_layer):
            layer_out = self.activ_list[i] \
                .act_forward(layer_out, self.w_list[i], self.b_list[i])
            self.y_list[i+1] = layer_out
        return layer_out

    def back_prop(self, target, conf):
        """
        do the actual back-propagation
        (refer to "Glossary" for notation & abbreviation)
        """
        # *_rate: learning rate (commonly referred to as alpha in literature)
        b_rate = conf.b_rate
        w_rate = conf.w_rate
        # momentum will speed up training, according to Geoff Hinton
        momentum = conf.momentum
        cur_c_d_y = self.cost.c_d_y(self.y_list[-1], target)
        for n in range(self.num_layer-1, -1, -1):
            cur_f = self.activ_list[n]
            cur_y = self.y_list[n+1]
            prev_y = self.y_list[n]
            cur_dw, cur_db, cur_c_d_y = cur_f.c_d_w_b_yn1(cur_c_d_y, cur_y, prev_y, self.w_list[n], is_c_d_yn1=n)
            if cur_dw is None:  # skip pooling layer
                continue
            #-------------#
            # update bias #
            #-------------#
            self.db_list[n] = momentum * self.db_list[n] + cur_db
            self.b_list[n] -= b_rate * self.db_list[n]
            #---------------#
            # update weight #
            #---------------#
            self.dw_list[n] = momentum * self.dw_list[n] + cur_dw
            self.w_list[n] -= w_rate * self.dw_list[n]


    def evaluate(self, batch_ipt, target):
        cur_net_out = self.net_act_forward(batch_ipt)
        cur_cost = sum(self.cost.act_forward(cur_net_out, target))
        cur_correct = 0
        if self.cost == cost_dict['CE']:
            cur_correct = (target.argmax(axis=1)==cur_net_out.argmax(axis=1)).sum()
        num_entry = batch_ipt.shape[0]
        return cur_cost/num_entry, cur_correct/num_entry
            

    def export_(self, path):
        """
        Save the checkpoint net configuration (*.npz)
        """
        info = {}
        info['w_list'] = self.w_list
        info['b_list'] = self.b_list
        info['activ_list'] = self.activ_list
        info['cost'] = self.cost
        info['epoch'] = self.epoch
        info['batch'] = self.batch
        np.savez(path, **info)

    
    def import_(self, path):
        """
        Load the checkpointing file for playing around the trained model
        """
        info = np.load(path)
        self.w_list = info['w_list']
        self.b_list = info['b_list']
        self.activ_list = info['activ_list']
        self.cost = info['cost']
        self.num_layer = len(self.w_list)
        self.dw_list = [None] * self.num_layer
        self.db_list = [None] * self.num_layer
        for i in range(self.num_layer):
            self.dw_list[i] = np.zeros(self.w_list[i].shape)
            self.db_list[i] = np.zeros(self.b_list[i].shape)
        self.y_list = [None] * (self.num_layer + 1)
        self.epoch = info['epoch']
        self.batch = info['batch']



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



#########################
#        NN flow        #
#########################
def net_train_main(yaml_model, args):
    """
    define main separately to facilitate unittest
    """
    # this is assuming that path of training data file is 
    # residing inside a dir named by the task name, and using '/' as delimiter
    db_subdir = yaml_model['obj_name']
    timestamp = strftime('%Y.%m.%d-%H.%M.%S')
    for f in _LOG_FILE.keys():
        _LOG_FILE[f] = _LOG_FILE[f].format(timestamp)
    
    #---------------------#
    #  data & net & conf  #
    #---------------------#
    data_set = Data(yaml_model, timestamp, profile=True)
    net = Net_structure(yaml_model)
    if args.partial_trained is not None:
        net.import_(args.partial_trained)
    print_to_file(_LOG_FILE['net'], net, type=None)

    conf = Conf(yaml_model)
    print_to_file(_LOG_FILE['conf'], conf)

    data_util.profile_net_conf(db_subdir, yaml_model, timestamp)
    #---------------------------#
    #  populate initial output  #
    #---------------------------#
    start_time = timeit.default_timer()
    cost_data = None    # populate into db in single db connection
    net_data = [[(-1,-1), data_set.target]] \
             #+ [[(0, 0), net.net_act_forward(data_set.data)]]

    #-----------------#
    #    main loop    #
    #-----------------#
    num_batch = data_set.data.shape[0] / conf.batch
    batch = 0
    indices = np.arange(data_set.data.shape[0])
    best_val_cost = sys.float_info.max
    best_val_correct = 0.

    for epoch in range(conf.num_epoch):
        # shuffle data
        if yaml_model['shuffle'] == True:
            np.random.shuffle(indices)
            data_set.data = data_set.data[indices]
            data_set.target = data_set.target[indices]
        net.epoch += 1
        cost_bat = 0.
        correct_bat = 0.
        epc_stride = 10
        for b, (bat_ipt, bat_tgt) in enumerate(data_set.get_batches(conf.batch)):
            batch += 1
            sys.stdout.write('\rbatch {}'.format(batch))
            sys.stdout.flush()
            net.batch += 1
            cur_cost_bat, cur_correct_bat = net.evaluate(bat_ipt, bat_tgt)
            printf('cur cost: {}', cur_cost_bat)
            cost_bat += cur_cost_bat
            correct_bat += cur_correct_bat
            net.back_prop(bat_tgt, conf)
        sys.stdout.write('\r')
        sys.stdout.flush()
        cost_bat /= num_batch
        correct_bat /= num_batch
        # validation & checkpointing
        _ = timeit.default_timer()
        cur_val_cost, cur_val_correct = net.evaluate(data_set.valid_d, data_set.valid_t)
        best_val_cost = (cur_val_cost<best_val_cost) and cur_val_cost or best_val_cost
        if cur_val_correct > best_val_correct:
            best_val_correct = cur_val_correct
            net.export_(yaml_model['checkpoint'])
        __ = timeit.default_timer()
        _TIME['checkpoint'] += (__ - _)

        if cost_data is None:
            cost_data = [[net.epoch, net.batch, cost_bat]]
        else:
            cost_data += [[net.epoch, net.batch, cost_bat]]
        #if (args.profile_output and net.epoch % epc_stride == 0) \
        #    or (net.epoch == conf.num_epoch):
        #    net_data += [[(net.epoch, net.batch), net.net_act_forward(data_set.data)]]
        printf('end of epoch {}, avg cost: {:.5f}, avg correct {:.3f}', net.epoch, cost_bat, correct_bat, type='TRAIN')
        printf("       cur validation accuracy: {:.3f}", cur_val_correct, type=None, separator=None)


    end_time = timeit.default_timer()
    printf('training took: {:.3f}', end_time-start_time)
    print_to_file(_LOG_FILE['net'], net, type=None)

    #--------------------#
    #  populate into db  #
    #--------------------#
    #start_time = timeit.default_timer()
    #data_util.profile_output_data(db_subdir, net_data, timestamp)
    #data_util.profile_cost(db_subdir, cost_data, timestamp)
    #end_time = timeit.default_timer()
    #printf('populate profiling data took: {:.3f}', end_time-start_time)
    #printf('time of checkpointing: {}', _TIME['checkpoint'])
    return end_time - start_time



if __name__ == '__main__':
    from pyspark import SparkContext
    import ec2.sc_glob as spark
    spark.sc = SparkContext(appName='ImageNet-dummy')
    args = parse_args()
    from stat_cnn.time import RUNTIME
    if args.yaml_model:
        yaml_model_full = 'yaml_model/{}.yaml'.format(args.yaml_model)
        yaml_model = yaml.load(open(yaml_model_full))
        tot_time = net_train_main(yaml_model, args)
        for k in RUNTIME.keys():
            printf('{} takes {:.3f}%', k, 100*RUNTIME[k]/tot_time, type='WARN')
