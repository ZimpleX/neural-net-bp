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
import argparse
from time import strftime

import pdb

np.random.seed(0)

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
    def __init__(self, layer_size_list, activ_list, cost_type):
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
        self.w_list = [np.random.rand(layer_size_list[i], layer_size_list[i+1]) - 0.5
                        for i in range(self.num_layer)]
        self.b_list = [np.random.rand(layer_size_list[i+1]) - 0.5 
                        for i in range(self.num_layer)]
        self.dw_list = [np.zeros((layer_size_list[i], layer_size_list[i+1]))
                        for i in range(self.num_layer)]
        self.db_list = [np.zeros(layer_size_list[i+1])
                        for i in range(self.num_layer)]
        self.activ_list = activ_list
        self.cost = cost_type
        # store the output of each layer
        self.y_list = [None] * (self.num_layer + 1)

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
        net_stat = ''
        for i in range(self.num_layer-1, -1, -1):
            net_stat += line_dash
            net_stat += 'layer {}: {} nodes\n'.format(i+1, self.w_list[i].shape[1])
            net_stat += line_dash
            net_stat += 'weight\n{}\nbias\n{}\n'.format(self.w_list[i], self.b_list[i])
        net_stat += line_dash
        net_stat += 'layer {}: {} nodes\n'.format(0, self.w_list[0].shape[0])
        net_stat += line_dash
        return net_stat

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
        # TODO: calculate c_d_ip (delta) instead of c_d_y
        #       separate it out to another loop
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
            # update bias #
            #-------------#
            temp = cur_c_d_y * cur_f.y_d_b(cur_y)
            temp = temp.reshape(hi_sz, cur_y.shape[-1]) # c_d_ip: batch_size x layer_nodes
            temp = np.sum(temp, axis=0)                 # vector of length layer_nodes
            # add momentum
            self.db_list[n] = momentum * self.db_list[n] + temp
            self.b_list[n] -= b_rate * self.db_list[n]
            #---------------#
            # update weight #
            #---------------#
            cur_c_d_y_exp = arr_util.expand_col(cur_c_d_y, self.w_list[n].shape[0])
            temp = cur_c_d_y_exp * cur_f.y_d_w(cur_y, prev_y)

            temp = temp.reshape([hi_sz] + list(self.w_list[n].shape))
            temp = np.sum(temp, axis=0)
            w_n = np.copy(self.w_list[n])
            # add momentum
            self.dw_list[n] = momentum * self.dw_list[n] + temp
            self.w_list[n] -= w_rate * self.dw_list[n]
            #---------------------------#
            # update derivative of cost #
            # w.r.t prev layer output   #
            #---------------------------#
            if n > 0:
                # don't update if prev layer is input layer
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
    parser.add_argument('-ptn', '--path_train', type=str, metavar='PATH_TRAIN',
            default=TRAIN_DATA, help='path to the training data set')
    parser.add_argument('-pts', '--path_test', type=str, metavar='PATH_TEST',
            default=TEST_DATA, help='path to the testing data set')
    parser.add_argument('--profile_cost', action='store_false',
            help='add this flag if you do NOT want to store cost profile')
    parser.add_argument('--profile_output', action='store_true',
            help='add this flag if you DO want to store the net output profile')
    return parser.parse_args()




#########################
#        NN flow        #
#########################
def net_train_main(args):
    """
    define main separately to facilitate unittest
    """
    timestamp = strftime('[%D-%H:%M:%S]')
    # this is assuming that path of training data file is 
    # residing inside a dir named by the task name, and using '/' as delimiter
    task_name = args.path_train.split('/')
    task_name = task_name[-2]

    assert len(args.struct) == len(args.activation) + 1
    data_set = Data(args.path_train, args.path_test)
    # auto correct shape of input / output layer of the ANN
    args.struct[0] = data_set.data.shape[1]
    args.struct[-1] = data_set.target.shape[1]
    net = Net_structure(args.struct, [activation_dict[n] for n in args.activation], cost_dict[args.cost])
    print('{}initial net\n{}{}\n'.format(line_star, line_star, net))
    conf = Conf(args.epoch, args.rate, args.inc_rate, args.dec_rate, args.momentum, 0.001)

    if (args.profile_cost): # store the conf of the ANN for this run
                            # could be identified by parse time
        data_util.profile_net_conf(task_name, args, timestamp)
        data_util.profile_raw_data_set(task_name, data_set, timestamp)

    # main training loop
    batch = 0
    for epoch in range(conf.num_epoch):
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
            cur_net_out = net.net_act_forward(bat_ipt)
            cost_bat += sum(Cost_sqr.act_forward(cur_net_out, bat_tgt))
            net.back_prop(bat_tgt, conf)
        if args.profile_cost:
            data_util.profile_cost(task_name, epoch, batch, cost_bat, timestamp)
        if args.profile_output and epoch % epc_stride == 0:
            data_util.profile_net_data(task_name, epoch, batch, net, data_set.data, timestamp)
        elif epoch == conf.num_epoch - 1:
            # populate final output data anyway
            data_util.profile_net_data(task_name, epoch, batch, net, data_set.data, timestamp)
        print('end of epoch {}, sum of cost over all batches: {}' \
                .format(epoch, cost_bat))

    print('{}final net\n{}{}\n'.format(line_star, line_star, net))
    print('final output {}\n'.format(net.y_list[2]))
    print(line_star*3)
    print(net.net_act_forward(data_set.test_d))




if __name__ == '__main__':
    args = parse_args()
    net_train_main(args)
