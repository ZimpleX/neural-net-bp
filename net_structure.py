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
from node_activity import *
from cost import *
from data_setup import *
from conf import *
import util.array_proc as arr_util
import util.data_proc as data_util
from logf.stringf import *
from logf.printf import *
from logf.filef import *
import argparse
from time import strftime
import timeit

import pdb

np.random.seed(100)

timestamp = strftime('%Y.%m.%d-%H.%M.%S')

_LOG_FILE = {'net': '{}-net'.format(timestamp),
            'conf': '{}-conf'.format(timestamp),}

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
    def __init__(self, layer_size_list, activ_list, cost_type, 
            w_range=INIT_RANGE['weight'], b_range=INIT_RANGE['bias']):
        """
        layer_size_list     let the num of nodes in each layer be Ni --> [N1,N2,...Nm]
                            including the input and output layer
        activ_list          list of node activation object, defining the 
                            activation function as well as derivatives
                            length of list = num of layers excluding input layer
        cost_type           the class representing the chosen cost function
        """
        # -1 to exclude the input layer
        self.num_layer = len(layer_size_list) - 1
        assert len(activ_list) == self.num_layer
        # *_list: list of 2D weight / 1D bias
        self.w_list = [(np.random.rand(layer_size_list[i], layer_size_list[i+1]) - 0.5) * w_range
                        for i in range(self.num_layer)]
        self.b_list = [(np.random.rand(layer_size_list[i+1]) - 0.5) * b_range
                        for i in range(self.num_layer)]
        self.dw_list = [np.zeros((layer_size_list[i], layer_size_list[i+1]))
                        for i in range(self.num_layer)]
        self.db_list = [np.zeros(layer_size_list[i+1])
                        for i in range(self.num_layer)]
        self.activ_list = activ_list
        self.cost = cost_type
        # store the output of each layer
        self.y_list = [None] * (self.num_layer + 1)
        # current epoch / batch: for debugging & logging
        self.epoch = -1
        self.batch = -1

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
            # size of higher dimension (e.g.: num of training tuples)
            hi_sz = cur_y.size / cur_y.shape[-1]
            #-------------#
            # update bias # NOTE: bias not applicable to conv layer
            #-------------#
            temp = cur_c_d_y * cur_f.y_d_b(cur_y)
            temp = temp.reshape(hi_sz, cur_y.shape[-1]) # c_d_ip: batch_size x layer_nodes
            temp = np.sum(temp, axis=0)                 # vector of length layer_nodes
            # add momentum
            self.db_list[n] = momentum * self.db_list[n] + temp
            self.b_list[n] -= b_rate * self.db_list[n]
            #---------------#
            # update weight # NOTE: conv layer: c_d_w = y_n_1 (*) flip(c_d_yn): with (*) representing conv operation
            #---------------#
            cur_c_d_y_exp = arr_util.expand_col(cur_c_d_y, self.w_list[n].shape[0])
            temp = cur_c_d_y_exp * cur_f.y_d_w(cur_y, prev_y)

            temp = temp.reshape([hi_sz] + list(self.w_list[n].shape))
            temp = np.sum(temp, axis=0)
            w_n = np.copy(self.w_list[n])   # TODO: probably not needing this: just move the cur_c_d_y computation up here
            # add momentum
            self.dw_list[n] = momentum * self.dw_list[n] + temp
            self.w_list[n] -= w_rate * self.dw_list[n]
            #---------------------------#
            # update derivative of cost #
            # w.r.t prev layer output   #
            #---------------------------#
            if n > 0:
                # don't update if prev layer is input layer
                #NOTE: for conv layer, compute as follow: c_d_yn1 = c_d_y (*) w, with (*) representing the conv operation
                d_chain = cur_f.yn_d_yn1(cur_y, w_n)
                cur_c_d_y = np.sum(cur_c_d_y_exp * d_chain, axis=-1)




###########################
#        Arg Parse        #
###########################
def parse_args():
    """
    accept cmd line options for specifying the ANN coefficient
    this func is also utilized by ./test/test.py
    """
    # TODO: add argument to enable choosing from several sets of default settings
    parser = argparse.ArgumentParser('settings for the ANN (accepting path of delimiter \'/\')')
    parser.add_argument('--struct', type=int, metavar='NET_STRUCT',
            default=STRUCT, nargs='+', help='specify the structure of the ANN (num of nodes in each layer)')
    parser.add_argument('--activation', type=str, metavar='NET_ACTIVATION',
            default=ACTIVATION, nargs='+', help='specify the activation of each layer',
            choices=activation_dict.keys())
    parser.add_argument('--cost', type=str, metavar='COST', 
            default=COST, help='specify the cost function',
            choices=cost_dict.keys())
    parser.add_argument('--w_range', type=int, metavar='W_RANGE', 
            default=INIT_RANGE['weight'], help='specify the range of weight initialization\nrandomized between [-W_RANGE, W_RANGE]')
    parser.add_argument('--b_range', type=int, metavar='B_RANGE',
            default=INIT_RANGE['bias'], help='specify the range of bias initialization\nrandomized between [-B_RANGE, B_RANGE]')
    parser.add_argument('-e', '--epoch', type=int, metavar='EPOCH',
            default=EPOCH, help='max num of iterations for training')
    parser.add_argument('-r', '--rate', type=float, metavar='RATE',
            default=LEARN_RATE, help='weight & bias update rate')
    parser.add_argument('--inc_rate', type=float, metavar='INC_RATE',
            default=INC_RATE, help='learning rate increment rate')
    parser.add_argument('--dec_rate', type=float, metavar='DEC_RATE',
            default=DEC_RATE, help='learning rate decrement rate')
    parser.add_argument('-m', '--momentum', type=float, metavar='MOMENTUM',
            default=MOMENTUM, help='momentum coefficient')
    parser.add_argument('-b', '--batch', type=int, metavar='BATCH',
            default=BATCH_SIZE, help='size of mini-batch')
    parser.add_argument('-dtbl', '--table_data', type=str, metavar='TABLE_DATA',
            default=TABLE_DATA, help='table name of training data')
    parser.add_argument('-ttbl', '--table_test', type=str, metavar='TABLE_TEST',
            default=TABLE_TEST, help='table name of test data')
    parser.add_argument('-szd', '--size_data', type=int, metavar='SIZE_DATA',
            default=SIZE_DATA, help='size of the data set (2^SIZE_DATA)')
    parser.add_argument('-szt', '--size_test', type=int, metavar='SIZE_TEST',
            default=SIZE_TEST, help='size of the test set (2^SIZE_TEST)')
    parser.add_argument('--profile_cost', action='store_false',
            help='add this flag if you do NOT want to store cost profile')
    parser.add_argument('--profile_output', action='store_true',
            help='add this flag if you DO want to store the net output profile')
    parser.add_argument('--log_verbose', type=int, metavar='LOG_V', 
            default=0, help='verbosity of logging: print the net to log fileevery LOG_V batch')
    return parser.parse_args()




#########################
#        NN flow        #
#########################
def net_train_main(args):
    """
    define main separately to facilitate unittest
    """
    # this is assuming that path of training data file is 
    # residing inside a dir named by the task name, and using '/' as delimiter
    db_subdir = ''
    
    assert len(args.struct) == len(args.activation) + 1
    data_set = Data(args.size_data, args.size_test, args.table_data, args.table_test, timestamp, profile=True, prof_subdir=db_subdir)
    # correct batch size if full batch is specified:
    if args.batch == -1:
        args.batch = data_set.data.shape[0]
    # auto correct shape of input / output layer of the ANN
    args.struct[0] = data_set.data.shape[1]
    args.struct[-1] = data_set.target.shape[1]
    net = Net_structure(args.struct, [activation_dict[n] for n in args.activation], cost_dict[args.cost], args.w_range, args.b_range)
    print_to_file(_LOG_FILE['net'], net, type=None)
    conf = Conf(args.epoch, args.rate, args.inc_rate, args.dec_rate, args.momentum, 0.001)

    print_to_file(_LOG_FILE['conf'], conf)

    if (args.profile_cost): # store the conf of the ANN for this run
                            # could be identified by parse time
        data_util.profile_net_conf(db_subdir, args, timestamp)

    # main training loop
    batch = 0
    # populate initial output: as to evaluate initial weight
    start_time = timeit.default_timer()
    cost_data = None    # populate into db in one run
    # profile init net data
    net_data = [[(-1,-1), data_set.target]] \
             + [[(0, 0), net.net_act_forward(data_set.data)]]
    num_batch = data_set.data.shape[0] / args.batch
    for epoch in range(conf.num_epoch):
        net.epoch = epoch + 1
        ######################
        #### Experimental ####
        #if epoch % 50 == 0 and epoch != 0:
        #    conf.w_rate *= 1.
        #    conf.b_rate *= 1.
        ######################
        cost_bat = 0.
        epc_stride = 10
        for b, (bat_ipt, bat_tgt) in enumerate(data_set.get_batches(args.batch)):
            batch += 1
            net.batch = batch
            if args.log_verbose > 0 and batch % args.log_verbose == 0:
                print_to_file(_LOG_FILE['net'], net, type=None) # logging
            cur_net_out = net.net_act_forward(bat_ipt)
            cost_bat += sum(Cost_sqr.act_forward(cur_net_out, bat_tgt))/args.batch
            net.back_prop(bat_tgt, conf)
            # TODO:
            # after some batches, evaluate validataion set, 
            # decrease learn rate if evaluation cost raises
        cost_bat /= num_batch
        if args.profile_cost:
            if cost_data == None:
                cost_data = [[net.epoch, net.batch, cost_bat]]
            else:
                cost_data += [[net.epoch, net.batch, cost_bat]]
        if (args.profile_output and net.epoch % epc_stride == 0) or (net.epoch == conf.num_epoch):
            net_data += [[(net.epoch, net.batch), net.net_act_forward(data_set.data)]]
        printf('end of epoch {}, sum of cost over all batches: {}', net.epoch, cost_bat, type='TRAIN')
    end_time = timeit.default_timer()
    printf('training took: {}', end_time-start_time)
    print_to_file(_LOG_FILE['net'], net, type=None)
    
    start_time = timeit.default_timer()
    data_util.profile_output_data(db_subdir, net_data, timestamp)
    data_util.profile_cost(db_subdir, cost_data, timestamp)
    end_time = timeit.default_timer()
    printf('populate profiling data took: {}', end_time-start_time)



if __name__ == '__main__':
    args = parse_args()
    net_train_main(args)
