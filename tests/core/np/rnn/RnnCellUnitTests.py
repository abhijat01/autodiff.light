import unittest

import core.np.Nodes as n
import core.np.rnn.Recurrent as rnn
import numpy as np
from core import log_at_info, debug
from tests.core.np.BaseTests import BaseComputeNodeTest


class SimpleRnnCellUnitTests(BaseComputeNodeTest):

    def setUp(self):
        x1 = np.array([1, 2, 1]).reshape((3, 1))
        self.var_map = {'x': x1}
        self.h = np.array([0.6, 0.2]).reshape((2, 1))
        w = np.array([[0.63315733, 0.51699569, 0.78251473, 0.94678789, 0.30939115],
                      [0.12741137, 0.67238871, 0.23514442, 0.50932127, 0.60643467],
                      [0.26004482, 0.02306102, 0.56403955, 0.32862147, 0.13988205],
                      [0.97815493, 0.66425931, 0.85988497, 0.13528022, 0.03943312]])
        wb = np.array([[0.5],
                       [-0.25],
                       [1.],
                       [-1.]])
        u = np.array([[0.39865366, 0.49334758, 0.29215267, 0.97590111, 0.68403036],
                      [0.03237844, 0.73579572, 0.49288022, 0.32059863, 0.69219668]])
        ub = np.array([[-1],
                       [.5]])
        self.w_param, self.wb_param = rnn.SharedParam(w), rnn.SharedParam(wb)
        self.u_param, self.ub_param = rnn.SharedParam(u), rnn.SharedParam(ub)

    def test_forward(self):
        input_x_node = n.VarNode('x')

        rnn_cell = rnn.RnnCell(input_x_node, None, self.w_param, self.wb_param,
                               self.u_param, self.ub_param, self.h)
        input_x_node.forward(self.var_map)
        y, h = rnn_cell.value()
        debug("[SimpleRnnCellTests.test_forward()] y = np.{}".format(repr(y)))
        debug("[SimpleRnnCellTests.test_forward()] h = np.{}".format(repr(h)))
        dely, delh = y * .1, h * .1
        rnn_cell.backward((dely, delh), self, self.var_map)
        grad_x = input_x_node.total_incoming_gradient()
        debug("[SimpleRnnCellTests.test_forward()] grad_x = np.{}".format(repr(grad_x)))

    def test_rnn_var_node(self):
        x = np.array([[1, 2, 1], [-1, 0, -.5]]).T
        x = x.reshape((3, 1, 2))
        x0_var = rnn.RnnVarNode(0, x)
        x1_var = rnn.RnnVarNode(1, x)
        np.testing.assert_equal(x0_var.value(), x[:, :, 0])
        np.testing.assert_equal(x1_var.value(), x[:, :, 1])
        debug("[SimpleRnnCellTests.test_rnn_var_node()] x0_var.value() = np.{}".format(repr(x0_var.value())))

    def test_2_seq_rnn(self):
        x = np.array([[1, 2, 1], [-1, 0, -.5]]).T
        x = x.reshape((3, 1, 2))
        x0_var = rnn.RnnVarNode(0, x)
        x1_var = rnn.RnnVarNode(1, x)
        cell1 = rnn.RnnCell(x0_var, None,self.w_param, self.wb_param, self.u_param,
                            self.ub_param, self.h )
        cell2 = rnn.RnnCell(x1_var, cell1,self.w_param, self.wb_param, self.u_param,
                            self.ub_param )
        x0_var.forward(self.var_map)
        x1_var.forward(self.var_map)
        y,h = cell2.value() 
        debug("[SimpleRnnCellTests.test_2_seq_rnn()] y = np.{}".format(repr(y)))
        debug("[SimpleRnnCellTests.test_2_seq_rnn()] h = np.{}".format(repr(h)))
        dely, delh = y * .1, h * .1
        cell2.backward((dely, None), self, var_map=self.var_map)
        wgrad  = self.w_param._total_incoming_gradient()
        debug("[SimpleRnnCellTests.test_2_seq_rnn()] wgrad = np.{}".format(repr(wgrad)))


if __name__ == '__main__':
    unittest.main()
