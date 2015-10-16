"""
this module defines all available training functions for generating the training data set
All training function will produce a target value between 0 and 1
"""
from math import exp, sin
from random import uniform
from net_structure import Net_structure
from cost import cost_dict
from node_activity import activation_dict
from numpy import *
import pdb

def trainingFunc(funcName):
    """
    first class function: return a closure of the actual training function

    argument:
        funcName        the description of how the training function should behave
    return:
        the actual training function
    """
    if funcName == "Sigmoid":
        """
        this is the very most suitable function for log neurons
        """
        def sigmoid(xList):
            xSum = reduce(lambda x1, x2: x1+x2, xList)
            return [ 1 / (1 + exp(-xSum)) ]
        return sigmoid
    elif funcName == "AttenSin1":
        def attenSin1(xList):
            assert len(xList) >= 1
            expPow = xList[0] * 0.2
            sinSum = reduce(lambda x1, x2: x1 + x2, xList) - xList[0]
            return [ exp(-abs(expPow)) * abs(sin(sinSum)) ]
        return attenSin1
    elif funcName == "Random":
        def rand(xList):
            return [ uniform(0, 1) ]
        return rand
    elif funcName == "AttenSin2":
        def attenSin2(xList):
            assert len(xList) >= 1
            expPow = xList[0] * 0.2
            sinSum = reduce(lambda x1, x2: x1 + x2, xList) - xList[0]
            return [ exp(-abs(expPow)) * sin(sinSum) ]
        return attenSin2
    elif funcName == "AttenSin3":
        def attenSin3(xList):
            sinSum = sum(xList)
            return [ exp(-0.06 * sinSum) * sin(sinSum) ]
        return attenSin3
    elif funcName == "ANN-bp":
        def forwardANN(xList, struct, activ_list, cost_type):
            net = Net_structure(struct, [activation_dict[n] for n in activ_list], cost_dict[cost_type])
            w_list = []
            b_list = []
            for l in range(len(struct) - 1):
                w_list += [(array(range(struct[l]*struct[l+1])).reshape(struct[l], struct[l+1]) + float(l)) / 100.]
                b_list += [(array(range(struct[l+1])) - float(l)) / 100.]

            net.set_w_b(w_list, b_list)
            return list(net.net_act_forward(array(xList)))
        return forwardANN
