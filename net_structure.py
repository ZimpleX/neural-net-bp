"""
define the general structure of the whole network, including:
    num of nodes in each layer;
    activate function of each layer;
    deritive of the active function (w.r.t weight / output of prev layer / bias)
"""

import numpy as np
import node_activity as act

class Net_structure:
    def __init__(self, layer_size_list, activ_list):
        """
        layer_size_list     let the num of nodes in each layer be Ni --> [N1,N2,...Nm]
                            including the input and output layer
        activ_list          list of node activation object, defining the 
                            activation function as well as derivatives
        """
        # num_layer is excluding the input layer
        self.num_layer = len(layer_size_list) - 1
        self.w_list = [np.zeros((layer_size_list[i], layer_size_list[i+1]))
                        for i in xrange(self.num_layer)]
        self.b_list = [np.zeros(layer_size_list[i+1]) for i in xrange(self.num_layer)]
        self.activ_list = activ_list

    def __str__(self):
        """
        print the value of weight and bias array, for each layer
        """
        net_stat = ''
        for i in xrange(self.num_layer-1, -1, -1):
            net_stat += '--------------------------\n'
            net_stat += 'layer {}: {} nodes\n'.format(i+1, self.w_list[i].shape[1])
            net_stat += '--------------------------\n'
            net_stat += 'weight\n{}\nbias\n{}\n'.format(self.w_list[i], self.b_list[i])
        net_stat += '--------------------------\n'
        net_stat += 'layer {}: {} nodes\n'.format(0, self.w_list[0].shape[0])
        net_stat += '--------------------------\n'
        return net_stat

    def act_forward(self, data):
        """
        action: 
            data = activity((data dot weight) + bias)
        argument:
            data    matrix of input data
                    columns = num of input nodes
                    rows = num of data pieces
        return:
            layer_out   the output from the output layer
        """
        layer_out = data
        for i in xrange(self.num_layer):
            layer_out = self.activ_list[i] \
                .act_forward(layer_out, w_list[i], b_list[i])
        return layer_out


if __name__ == "__main__":
    ns = Net_structure([2,3,4], [act.Node_linear] * 3)
    print(ns)
    print act.Node_sigmoid.act_forward(np.array(range(4)).reshape(2,2), np.array(range(6)).reshape(2,3), np.array([3,4,5]))

