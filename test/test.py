from cost import *
from net_structure import *
from node_activity import *
from conf import *
from data_setup import *
import unittest as ut
from numpy import *
from numpy.testing import *

ACT_FORWARD_FAIL='net act forward failed'

class Test_deriv(ut.TestCase):
    def test_lin3_lin2_single_tuple(self, verbose=True):
        net = Net_structure([3,2], [Node_linear], Cost_sqr)
        net.set_w_b([array(range(6)).reshape(3,2)], [zeros((1,2))])
        if verbose:
            print('{}initial net\n{}{}\n'.format(line_star, line_star, net))
        data = array([[11,22,33], [-1, 2, 82]])
        net_op = array([[176., 242.], [332., 415.]])
        target = array([[5.,4.], [2., 5.]])
        test_net_op = net.net_act_forward(data)
        print('net forward output {}\n'.format(test_net_op))
        assert_array_equal(net_op, test_net_op, ACT_FORWARD_FAIL)
        self.assertSequenceEqual(net_op.shape, test_net_op.shape, ACT_FORWARD_FAIL)
        assert_array_equal(net.cost.act_forward(net_op, target), [42942.5, 138500.])
        assert_array_equal(net.cost.c_d_y(net_op, target), array([[171., 238.], [330., 410.]]))
        assert_array_equal(net.activ_list[0].yn_d_yn1(net.y_list[1], net.w_list[0]), array([net.w_list[0], net.w_list[0]]))
        #if verbose:
        #    print('{}net after ff\n{}{}\n'.format(line_star, line_star, net))


if __name__ == "__main__":
    # ut.main()
    suite = ut.TestLoader().loadTestsFromTestCase(Test_deriv)
    ut.TextTestRunner(verbosity=2).run(suite)
