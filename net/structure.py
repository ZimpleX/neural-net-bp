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
from net.node_activity import *
import conv.util
from net.cost import *
from net.data_setup import *
from net.conf import *
import util.data_proc as data_util
from logf.stringf import *
from logf.printf import *
from logf.filef import *
from time import strftime
import timeit
import sys
import copy
import os
from functools import reduce
from stat_cnn.mem_usage import *

import conv.conv_layer
import conv.pool_layer

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
    def __init__(self, yaml_model, slide_method, sc):
        """
        layer_size_list     let the num of nodes in each layer be Ni --> [N1,N2,...Nm]
                            including the input and output layer
        activ_list          list of node activation object, defining the 
                            activation function as well as derivatives
                            length of list = num of layers excluding input layer
        cost_type           the class representing the chosen cost function
        """
        if yaml_model is None:
            self.w_list = None
            self.b_list = None
            return

        if sc is None:
            num_exe = 0
        else:
            # num_exe = sc.defaultParallelism
            # TODO: change this --> auto get the number of cores
            num_exe = 8
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
        for i in range(len(yaml_model['network'])):
            l = yaml_model['network'][i]
            init_wt = ('init_wt' in l) and l['init_wt'] or 0.0
            cur_chn = l['num_channels']
            act_func = activation_dict[l['type']]
            if l['type'] == 'CONVOLUTION' or l['type'] == 'MAXPOOL':
                if prev_img is None:
                    prev_img = (yaml_model['input_image_size_y'], yaml_model['input_image_size_x'])
                chan = l['num_channels']
                kern = l['kernel_size']
                w_shape = (prev_chn, cur_chn, kern, kern)
                act_init = (l['type']=='MAXPOOL') and [chan, kern] or []
                act_init += [l['stride'], l['padding']]
                act_init += [slide_method]
                SparkMeta = {}
                SparkMeta['num_executor'] = num_exe
                if yaml_model['network'][i+1]['type'] not in ['CONVOLUTION', 'MAXPOOL']:
                    SparkMeta['conn_to_FC'] = True
                else:
                    SparkMeta['conn_to_FC'] = False
                if i == 0:
                    SparkMeta['first_conv'] = True
                else:   
                    SparkMeta['first_conv'] = False
                act_init += [SparkMeta]
                self.activ_list[idx] = act_func(*act_init)
                prev_img = conv.util.ff_next_img_size(prev_img, kern, l['padding'], l['stride'])
            else:
                if prev_img is not None:
                    prev_chn = prev_chn * prev_img[0] * prev_img[1]
                    prev_img = None
                w_shape = (prev_chn, cur_chn)
                self.activ_list[idx] = act_func()
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


    def flattern_w_b(self):
        """
        flattern the net's w & b.

        every feature map in the conv layer gets an index,
        the whole weight between FC layers gets an index.
        
        The second tuple is for mapping back. 
        e.g.,
            the [i][j] feature map in layer k gets an index (k,(i,j))
        """
        w_flatterned = []
        b_flatterned = []
        for i,w in enumerate(self.w_list):
            # TODO: skip pooling layers
            if type(self.activ_list[i]) is conv.pool_layer.Node_pool:
                continue
            if len(w.shape) == 4:
                d0 = w.shape[0]
                d1 = w.shape[1]
                w_flatterned += [(w[ii//d1][ii%d1], (i, (ii//d1, ii%d1))) for ii in range(d0*d1)]
            else:
                w_flatterned += [(w, (i,))]

        for j,b in enumerate(self.b_list):
            if type(self.activ_list[i]) is conv.pool_layer.Node_pool:
                continue
            b_flatterned += [(b, (j,))]
        
        return w_flatterned, b_flatterned

    def eval_delta_w_b(self, orig_flattern_w, orig_flattern_b, cur_flattern_w, cur_flattern_b):
        """
        evaluate how much w (feamap) & b has changed through these epochs.
        """
        eval_delta_w = []
        for j,w in enumerate(cur_flattern_w):
            dw = (w[0] - orig_flattern_w[j][0]).reshape(1, -1)
            _ = reduce(lambda _1,_2: _1+_2, map(lambda x: x*x, dw))
            eval_delta_w += [(_, w[1])]
        eval_delta_b = []
        for i,b in enumerate(cur_flattern_b):
            db = b[0] - orig_flattern_b[i][0]
            _ = reduce(lambda _1,_2: _1+_2, map(lambda x: x*x, db))
            eval_delta_b += [(_, b[1])]
        return eval_delta_w, eval_delta_b


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

    def net_act_forward(self, data, sc):
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
                .act_forward(layer_out, self.w_list[i], self.b_list[i], sc=sc)
            self.y_list[i+1] = layer_out
        return layer_out

    def back_prop(self, target, conf, sc):
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
            cur_dw, cur_db, cur_c_d_y = cur_f.c_d_w_b_yn1(cur_c_d_y, cur_y, prev_y, self.w_list[n], is_c_d_yn1=n, sc=sc)
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


    def evaluate(self, batch_ipt, target, mini_batch=0, sc=None, eval_details=False, eval_name='null'):
        """
        mini_batch:     if the evaluation set is large, then evaluate all at a time may cause out of memory error,
                        especially when the DCNN has many layers.
                        in case of that, evaluate batch by batch.
                        mini_batch=0 by default --> evaluate all at once.
        """
        if batch_ipt is None or target is None:
            return 0.,0.
        num_entry = batch_ipt.shape[0]
        cur_cost = 0.
        mini_batch = (mini_batch <= 0) and num_entry or mini_batch
        cur_correct = 0.
        if eval_details:
            num_out_cat = target.shape[1]
            _attr = ['name', 'index', 'is_correct'] + ['cat{}'.format(i) for i in range(num_out_cat)]
            _type = ['TEXT', 'INTEGER', 'INTEGER'] + ['REAL']*num_out_cat
            _db_name = 'eval_out_prob.db'
            tot_net_out = None
            tot_compare = None
           
        for k in range(0, num_entry, mini_batch):
            cur_batch = batch_ipt[k:(k+mini_batch)]
            cur_target = target[k:(k+mini_batch)]
            cur_net_out = self.net_act_forward(cur_batch, sc)
            cur_cost += sum(self.cost.act_forward(cur_net_out, cur_target))
            if self.cost == cost_dict['CE']:
                _compare = (cur_target.argmax(axis=1)==cur_net_out.argmax(axis=1))
                cur_correct += (_compare).sum()
                if eval_details:
                    # populate output of the CNN for detailed inspection
                    cur_net_out.sort(axis=1)
                    _compare = _compare[..., np.newaxis].astype(np.int)
                    if tot_compare is None:
                        tot_compare = copy.deepcopy(_compare)
                    else:
                        tot_compare = np.concatenate((tot_compare, _compare), axis=0)
                    if tot_net_out is None:
                        tot_net_out = copy.deepcopy(cur_net_out)
                    else:
                        tot_net_out = np.concatenate((tot_net_out, cur_net_out), axis=0)
        if eval_details:
            # TODO: should also set a limit for one-time populating output array.
            _idx = (np.arange(num_entry))[..., np.newaxis]
            from db_util.basic import populate_db
            _pair = np.concatenate((tot_compare, tot_net_out), axis=1)
            _field = ','.join(['i8'] + ['f8']*num_out_cat)
            _pair.view(_field).sort(order=['f0', 'f{}'.format(num_out_cat)], axis=0)
            populate_db(_attr, _type, (eval_name,), _idx, _pair, db_name=_db_name)
        return cur_cost/num_entry, cur_correct/num_entry
            

    def export_(self, yaml_model, sync_script=None):
        """
        Save the checkpoint net configuration (*.npz)
        """
        # TODO: maybe support sync to remote?
        path = yaml_model['checkpoint']
        info = {}
        info['w_list'] = self.w_list
        info['b_list'] = self.b_list
        info['yaml_model'] = yaml_model
        info['cost'] = self.cost
        info['epoch'] = self.epoch
        info['batch'] = self.batch
        np.savez(path, **info)
        printf('save checkpoint file: {}', path)

    
    def import_(self, path, slide_method='slide_serial', sc=None):
        """
        Load the checkpointing file for playing around the trained model
        """
        info = np.load(path)
        yaml_model = info['yaml_model'].reshape(1)[0]
        self.__init__(yaml_model, slide_method, sc)
        if self.w_list is not None:
            # the case of partial trained chkpt using new yaml_model
            # this is to ensure that the yaml_model in args and that in chkpt matches
            for i,w in enumerate(info['w_list']):
                assert w.shape == self.w_list[i].shape
        if self.b_list is not None: 
            for i,b in enumerate(info['b_list']):
                assert b.shape == self.b_list[i].shape

        self.w_list = info['w_list']    # probably need to convert to list
        self.b_list = info['b_list']
        self.cost = info['cost'].reshape(1)[0]  # probably no need for this
        self.num_layer = len(self.w_list)
        self.dw_list = [None] * self.num_layer
        self.db_list = [None] * self.num_layer
        for i in range(self.num_layer):
            self.dw_list[i] = np.zeros(self.w_list[i].shape)
            self.db_list[i] = np.zeros(self.b_list[i].shape)
        self.y_list = [None] * (self.num_layer + 1)
        self.epoch = info['epoch']
        self.batch = info['batch']





#########################
#        NN flow        #
#########################
def net_train_main(yaml_model, args, sc, old_net=None):
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
    conf = Conf(yaml_model)
    if old_net is None:
        net = Net_structure(yaml_model, args.slide_method, sc)
        if args.partial_trained is not None:
            net.import_(args.partial_trained, args.slide_method, sc)
        print_to_file(_LOG_FILE['net'], net, type=None)
        print_to_file(_LOG_FILE['conf'], conf)
    else:
        net = old_net

    orig_flattern_w, orig_flattern_b = copy.deepcopy(net.flattern_w_b())

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
            data_set.shuffle()
        net.epoch += 1
        cost_bat = 0.
        correct_bat = 0.
        for b, (bat_ipt, bat_tgt) in enumerate(data_set.get_batches(conf.batch)):
            batch += 1
            net.batch += 1
            cur_cost_bat, cur_correct_bat = net.evaluate(bat_ipt, bat_tgt, sc=sc)
            printf('cur cost: {:.4f}', cur_cost_bat)
            printf('epoch {}, batch {}', epoch, batch, type="WARN")
            cost_bat += cur_cost_bat
            correct_bat += cur_correct_bat
            net.back_prop(bat_tgt, conf, sc)
            
        sys.stdout.write('\r')
        sys.stdout.flush()
        cost_bat /= num_batch
        correct_bat /= num_batch
        # validation & checkpointing
        _ = timeit.default_timer()
        #try:
        cur_val_cost, cur_val_correct = net.evaluate(data_set.valid_d, data_set.valid_t, mini_batch=50, sc=sc)
        best_val_cost = (cur_val_cost<best_val_cost) and cur_val_cost or best_val_cost
        if cur_val_correct > best_val_correct:
            best_val_correct = cur_val_correct
            net.export_(yaml_model)
        __ = timeit.default_timer()
        _TIME['checkpoint'] += (__ - _)

        cost_data  = [[net.epoch, net.batch, cost_bat, correct_bat, cur_val_cost, cur_val_correct]]
        data_util.profile_cost(db_subdir, cost_data, timestamp)
   
        #if (args.profile_output and net.epoch % epc_stride == 0) \
        #    or (net.epoch == conf.num_epoch):
        #    net_data += [[(net.epoch, net.batch), net.net_act_forward(data_set.data)]]
        printf('end of epoch {}, avg cost: {:.5f}, avg correct {:.3f}', net.epoch, cost_bat, correct_bat, type='TRAIN')
        printf("       cur validation accuracy: {:.3f}", cur_val_correct, type=None, separator=None)


    end_time = timeit.default_timer()
    printf('training took: {:.3f}', end_time-start_time)
    # print_to_file(_LOG_FILE['net'], net, type=None)
    cur_flattern_w, cur_flattern_b = net.flattern_w_b()
    eval_dw, eval_db = net.eval_delta_w_b(orig_flattern_w, orig_flattern_b, cur_flattern_w, cur_flattern_b)

    return end_time - start_time, eval_dw, eval_db, cur_flattern_w, cur_flattern_b


