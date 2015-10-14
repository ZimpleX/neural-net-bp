"""
define the general structure of the whole network, including:
    num of nodes in each layer;
    activate function of each layer;
    deritive of the active function (w.r.t weight / output of prev layer / bias)


TODO:
    <del>convert to mini-batch or online training</del>
    sum Cost function w.r.t all examples in one mini-batch
    when obtaining delta, divide by size of mini-batch
    add momentum / adapt learning rate
    add gradient checking using finite difference
"""

import numpy as np
from node_activity import *
from cost import *
from data_setup import *
from conf import *
import util.array_proc as arr_util
import argparse

import pdb


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

        self.w_list = [np.random.rand(layer_size_list[i], layer_size_list[i+1]) - 0.5
                        for i in range(self.num_layer)]
        self.b_list = [np.random.rand(layer_size_list[i+1]) - 0.5 for i in range(self.num_layer)]
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
                    n-1 = columns: num of input nodes
                    n = rows: num of data pieces
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
        """
        b_rate = conf.b_rate
        w_rate = conf.w_rate
        cur_c_d_y = self.cost.c_d_y(self.y_list[-1], target)
        for n in range(self.num_layer-1, -1, -1):
            cur_f = self.activ_list[n]
            cur_y = self.y_list[n+1]
            prev_y = self.y_list[n]
            # size of higher dimension (e.g.: num of training tuples)
            hi_sz = cur_y.size / cur_y.shape[-1]
            # update bias
            temp = cur_c_d_y * cur_f.y_d_b(cur_y)
            temp = temp.reshape(hi_sz, cur_y.shape[-1])
            temp = np.sum(temp, axis=0)
            if __debug__:
                print('bias update\n{}\n'.format(temp))
            self.b_list[n] -= b_rate * temp
            # update weight
            cur_c_d_y_exp = arr_util.expand_col(cur_c_d_y, self.w_list[n].shape[0])
            temp = cur_c_d_y_exp * cur_f.y_d_w(cur_y, prev_y)
            #pdb.set_trace()

            temp = temp.reshape([hi_sz] + list(self.w_list[n].shape))
            temp = np.sum(temp, axis=0)
            if __debug__:
                print('weight update\n{}\n'.format(temp))
            w_n = np.copy(self.w_list[n])
            self.w_list[n] -= w_rate * temp
            # update derivative of cost w.r.t prev layer output
            if n > 0:
                # don't update if prev layer is input layer
                d_chain = cur_f.yn_d_yn1(cur_y, w_n)
                if __debug__:
                    print('layer {} d yn d yn-1 \n{}\n'.format(n, d_chain))
                cur_c_d_y = np.sum(cur_c_d_y_exp * d_chain, axis=-1)


def parse_args():
    epc_default = None
    wr_default = 0.001
    br_default = 0.001
    batch_def = 1
    ptrain_def = 'train_data/AttenSin2/3_04'
    ptest_def = 'train_data/AttenSin2/3_03'
    if __debug__:
        epc_default = 3
    else:
        epc_default = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, metavar='EPOCH',
            default=epc_default, help='max num of iterations for trainging')
    parser.add_argument('-wr', '--w_rate', type=float, metavar='W_RATE',
            default=wr_default, help='weight update rate')
    parser.add_argument('-br', '--b_rate', type=float, metavar='B_RATE',
            default=br_default, help='bias update rate')
    parser.add_argument('-bt', '--batch', type=int, metavar='BATCH',
            default=batch_def, help='size of mini-batch')
    parser.add_argument('-ptn', '--path_train', type=str, metavar='PATH_TRAIN',
            default=ptrain_def, help='path to the training data set')
    parser.add_argument('-pts', '--path_test', type=str, metavar='PATH_TEST',
            default=ptest_def, help='path to the testing data set')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    if __debug__:
        print('{}{}debug on{}{}'.format(line_ddash, line_ddash, line_ddash, line_ddash))
    net = Net_structure([3,10,1], [Node_sigmoid, Node_linear], Cost_sqr)
    #net = Net_structure([3,1], [Node_sigmoid], Cost_sqr)
    print('{}initial net\n{}{}\n'.format(line_star, line_star, net))
    data_set = Data(args.path_train, args.path_test, [TARGET, INPUT, INPUT, INPUT])
    conf = Conf(args.epoch, args.w_rate, args.b_rate, 0.001)
    print(data_set)
    for epoch in range(conf.num_epoch):
        cost_bat = 0.
        for b, (bat_ipt, bat_tgt) in enumerate(data_set.get_batches(args.batch)):
            cur_net_out = net.net_act_forward(bat_ipt)
            cost_bat += sum(Cost_sqr.act_forward(cur_net_out, bat_tgt))
            stride = None
            if __debug__:
                stride = 1
            else:
                stride = 10
            if b % stride == 0:
                if __debug__:
                    print('\t{}\tepoch {} batch {}\n\tcost {}\n\t{}\n' \
                            .format(line_star, epoch, b, Cost_sqr.act_forward(cur_net_out, bat_tgt), line_star))
                    print('{}output of all layers\n{}\n{}'.format(line_ddash, net.y_list, line_ddash))
            net.back_prop(bat_tgt, conf)
            print('{}cur net\n{}{}\n'.format(line_star, line_star, net))
        print('{}end of epoch {}, sum of cost over all batches: {}\n{}' \
                .format(line_ddash, epoch, cost_bat, line_ddash))

    print('{}final net\n{}{}\n'.format(line_star, line_star, net))
    print('final output {}\n'.format(net.y_list[2]))
    print(line_star*3)
    print(net.net_act_forward(data_set.test_d))
    if __debug__:
        print('debug {}'.format(__debug__))
        print('num epoches {}\n'.format(conf.num_epoch))
