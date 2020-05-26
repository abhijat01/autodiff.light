import unittest
import numpy as np

import core.np.Nodes as node
import math

import core.np.Optimization as optim
from core.np.Activations import SigmoidNode
from core import debug, info, log_at_info
from . import BaseComputeNodeTest

from core.np.Loss import L2DistanceSquaredNorm


class BasicNetworkNoActivation(BaseComputeNodeTest):
    def test_single_step(self):
        w_node = node.VarNode('w')
        x_node = node.VarNode('x')
        ya_node = node.VarNode('y_a')
        b_node = node.VarNode('b')

        w = np.array([[1, 3, 0], [0, 1, -1]])
        x = (np.array([[1, -1, 2]])).T
        b = np.array([[-2, -3]]).T
        y_act = np.array([[1, 2]]).T

        var_map = {'w': w, 'x': x, 'y_a': y_act, 'b': b}

        wx_node = node.MatrixMultiplication(w_node, x_node)
        sum_node = node.MatrixAddition(wx_node, b_node)
        l2_node = L2DistanceSquaredNorm(sum_node, ya_node)
        w_node.forward(var_map)
        x_node.forward(var_map)
        b_node.forward(var_map)
        ya_node.forward(var_map)
        l2norm = l2_node.value()
        info("L2Norm: {}".format(l2norm))
        y_p = w @ x + b
        y_del = y_p - y_act
        expected = np.sum(np.square(y_del)) / y_del.size
        debug("l2norm:{}".format(l2_node))
        self.assertEqual(expected, l2norm)
        l2_node.backward(1.0, self, var_map)
        w_grad = w_node.total_incoming_gradient()
        b_grad = b_node.total_incoming_gradient()
        debug("----- w grad ------")
        debug(w_grad)
        debug("-------------------")
        debug("----- b grad ------")
        debug(b_grad)
        debug("-------------------")
        l2_node.reset_network_back()
        self.assertIsNone(w_node.total_incoming_gradient())
        l2_node.backward(1.0, self, var_map)
        w_grad = w_node.total_incoming_gradient()
        b_grad = b_node.total_incoming_gradient()
        debug("----- w grad ------")
        debug(w_grad)
        debug("-------------------")
        debug("----- b grad ------")
        debug(b_grad)
        debug("-------------------")

    def test_network_optimizer(self):
        w_node = node.VarNode('w', True)
        x_node = node.VarNode('x')
        ya_node = node.VarNode('y_a')
        b_node = node.VarNode('b', True)
        start_nodes = [w_node, x_node, b_node, ya_node]

        w = np.array([[1, 3, 0], [0, 1, -1]])
        x = (np.array([[1, -1, 2]])).T
        b = np.array([[-2, -3]]).T
        y_act = np.array([[1, 2]]).T
        var_map = {'w': w, 'x': x, 'y_a': y_act, 'b': b}

        wx_node = node.MatrixMultiplication(w_node, x_node)
        sum_node = node.MatrixAddition(wx_node, b_node)
        l2_node = L2DistanceSquaredNorm(sum_node, ya_node)
        optimizer = optim.OptimizerIterator(start_nodes, l2_node)
        log_at_info()
        for _ in range(500):
            loss = optimizer.step(var_map, 1.0)
        info("Final loss:{}".format(loss))
        self.assertTrue(math.fabs(loss) < 1e-25)


class BasicNetworkSigmoid(BaseComputeNodeTest):
    def test_sigmoid_node(self):
        x_node = node.VarNode('x')
        x = (np.array([[1, -1, 2]])).T
        var_map = {'x': x}
        sigmoid = SigmoidNode(x_node)
        x_node.forward(var_map)
        value = sigmoid.value()
        expected_value = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(expected_value, value)
        debug(value)
        sigmoid.backward(np.ones_like(value), self, var_map)
        grad = x_node.total_incoming_gradient()
        expected_grad = expected_value * (1 - expected_value)
        debug(grad)
        np.testing.assert_array_almost_equal(expected_grad / expected_grad.size, grad)


if __name__ == '__main__':
    unittest.main()
