"""
NOTE:
    invoke this script from root dir
"""
from logf.printf import *
from logf.filef import *
from net.cost import *
from net.structure import *
from net.node_activity import *
from net.conf import *
from net.data_setup import *
import unittest as ut
from numpy import *
from numpy.testing import *
from util.profile_data import *
from util.data_generator import *
import db_util as db
import os
import argparse
import sqlite3
import yaml

import conv.conv as conv

import pdb

ACT_FORWARD_FAIL='net act forward failed'
args = None

class Test_deriv(ut.TestCase):
    def test_lin3_lin2_single_tuple(self, verbose=True):
        net = Net_structure([3,2], [Node_linear], Cost_sqr)
        net.set_w_b([array(range(6)).reshape(3,2)], [zeros((1,2))])
        if verbose:
            printf('INITIAL NET')
            printf(net, type=None, separator=None)
        data = array([[11,22,33], [-1, 2, 82]])
        net_op = array([[176., 242.], [332., 415.]])
        target = array([[5.,4.], [2., 5.]])
        test_net_op = net.net_act_forward(data)
        printf('net forward output')
        printf(test_net_op, type=None, separator=None)
        assert_array_equal(net_op, test_net_op, ACT_FORWARD_FAIL)
        self.assertSequenceEqual(net_op.shape, test_net_op.shape, ACT_FORWARD_FAIL)
        assert_array_equal(net.cost.act_forward(net_op, target), [42942.5, 138500.])
        assert_array_equal(net.cost.c_d_y(net_op, target), array([[171., 238.], [330., 410.]]))
        assert_array_equal(net.activ_list[0].yn_d_yn1(net.y_list[1], net.w_list[0]), array([net.w_list[0], net.w_list[0]]))

class Test_util(ut.TestCase):
    def test_populate_db(self):
        populate_db(['a', 'b', 'c'], ['TEXT', 'INTEGER', 'REAL'], ['yo'], array(range(6)).reshape(3,2), db_path='data/', db_name='unittest.db')
        os.remove('data/unittest.db')

class Test_net(ut.TestCase):
    def test_3_layer_bp(self):
        # generate data if not exist
        dg_args = argparse.ArgumentParser()
        dg_args.function = 'ANN-bp'
        dg_args.data_size = 12
        dg_args.struct = args.struct
        dg_args.activation = args.activation
        dg_args.cost = args.cost
        p_data = dataGeneratorMain(dg_args)

        args.path_train = p_data + '08'
        args.path_test = p_data + '04'
        net_train_main(args)
        
class Test_db_util(ut.TestCase):
    def test_db_basic(self):
        print()
        db_name = 'cool.db'
        table_name = 'stuff'
        db_path = './unittest/db/basic/'
        db.basic.populate_db(['this','is','cool'], ['INTEGER','TEXT','REAL'], 
            0, [['zero',0.],['two',0.2],['one',0.1]],
            db_path=db_path, db_name=db_name, table_name=table_name)
        db.basic.sanity_db('is', ['zero'], 'stuff', db_name=db_name, db_path=db_path)
        db.basic.sanity_db(['is'], 'zero', 'stuff', db_name=db_name, db_path=db_path)
        db.basic.sanity_db('shhhh', 'zero', 'stuff', db_name=db_name, db_path=db_path)
        printf('CHECK THE DB YOURSELF TO SEE IF TEST IS PASSED!', type='WARN', separator='*')
    def test_db_interact(self):
        print()
        db_path = './profile_data/Sin_in-1-out-1/'
        db_name = 'ann.db'
        meta_table = 'meta|ann'
        data_table = 'profile_cost|ann'
        db.interact.db_control_dim(meta_table, data_table, 'learn_rate', db_path=db_path, db_name=db_name)
        printf('CHECK THE DB YOURSELF TO SEE IF TEST IS PASSED!', type='WARN', separator='*')
    def test_db_sanity_last_n(self):
        print()
        db_path = './profile_data/Sin_in-1-out-1/'
        db_name = 'ann.db'
        db.interact.sanity_last_n_commit('meta|ann',db_name=db_name, db_path=db_path)
        db.interact.sanity_last_n_commit(db_name=db_name, db_path=db_path)
        printf('CHECK THE DB YOURSELF TO SEE IF TEST IS PASSED!', type='WARN', separator='*')
        
       


class Test_conv(ut.TestCase):
    def test_conv(self):
        print()
        image_path_gs = './test/grayscale.jpg'
        output_path_gs = './test/output_grayscale.jpg'
        image_path_rgb = './test/rgb.jpg'
        output_path_rgb = './test/output_rgb.jpg'
        kernel_core = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        conv_layer = conv.Node_conv

        from PIL import Image
        printf('conv grayscale')
        layer_img_gs = np.asarray(Image.open(image_path_gs))
        Y,X = layer_img_gs.shape
        C = 1
        layer_img_gs = layer_img_gs.reshape(Y,X,C)
        layer_img_gs = layer_img_gs.transpose((2,0,1)).reshape(1,C,Y,X)
        kernel_gs = np.zeros((1,1,3,3))
        kernel_gs[0,0,:,:] = kernel_core

        output_img_gs = conv_layer.act_forward(layer_img_gs, np.swapaxes(kernel_gs,0,1), 1, 1)
        output_img_gs = output_img_gs.reshape(C,Y,X).transpose(1,2,0)
        output_img_gs = output_img_gs.reshape(Y,X)
        Image.fromarray(np.uint8(output_img_gs.clip(0,255))).save(output_path_gs)
        
        printf('conv rgb')
        layer_img_rgb = np.array(Image.open(image_path_rgb))
        Y,X,C = layer_img_rgb.shape
        layer_img_rgb = layer_img_rgb.transpose((2,0,1)).reshape(1,C,Y,X)
        kernel_rgb = np.zeros((3,3,3,3))
        kernel_rgb[0,0,:,:] = kernel_core
        kernel_rgb[1,1,:,:] = kernel_core
        kernel_rgb[2,2,:,:] = kernel_core
        
        output_img_rgb = conv_layer.act_forward(layer_img_rgb, np.swapaxes(kernel_rgb,0,1), 1, 1)
        output_img_rgb = output_img_rgb.reshape(C,Y,X).transpose(1,2,0)
        Image.fromarray(np.uint8(output_img_rgb.clip(0,255))).save(output_path_rgb)



if __name__ == "__main__":
    # accept args the same way as the main in net.structure.py
    args = parse_args()
    # ut.main()
    #suite = ut.TestLoader().loadTestsFromTestCase(Test_deriv)
    suite = ut.TestLoader().loadTestsFromTestCase(Test_conv)
    ut.TextTestRunner(verbosity=2).run(suite)
