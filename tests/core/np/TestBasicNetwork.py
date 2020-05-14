import unittest
import numpy as np

import core.np.Nodes as node
import math

import core.np.Optimization
from  core.np.Activations import SigmoidNode
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
        w_node.forward(var_map, None, self)
        x_node.forward(var_map, None, self)
        b_node.forward(var_map, None, self)
        ya_node.forward(var_map, None, self)
        l2norm = l2_node.value(var_map)
        info("L2Norm: {}".format(l2norm))
        y_p = w @ x + b
        y_del = y_p - y_act
        expected = np.sum(np.square(y_del))/y_del.size
        debug("l2norm:{}".format(l2_node))
        self.assertEqual(expected, l2norm)
        l2_node.backward(1.0, self, var_map, " ")
        w_grad = w_node.grad_value()
        b_grad = b_node.grad_value()
        debug("----- w grad ------")
        debug(w_grad)
        debug("-------------------")
        debug("----- b grad ------")
        debug(b_grad)
        debug("-------------------")
        l2_node.reset_network_back()
        self.assertIsNone(w_node.grad_value())
        l2_node.backward(1.0, self, var_map, " ")
        w_grad = w_node.grad_value()
        b_grad = b_node.grad_value()
        debug("----- w grad ------")
        debug(w_grad)
        debug("-------------------")
        debug("----- b grad ------")
        debug(b_grad)
        debug("-------------------")

    def test_optimizer_step(self):
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

        for start_node in start_nodes:
            start_node.forward(var_map, None, self)

        l2_node.backward(1.0, self, var_map, " ")
        loss = l2_node.value(var_map)
        debug("Loss:{}".format(loss))

        def optimizer(w, grad, node_local_storage={}):
            return w - 0.01 * grad

        debug("W before step")
        debug(var_map['w'])
        l2_node.optimizer_step(optimizer, var_map)
        debug("W after step")
        debug(var_map['w'])

        losses = []
        for i in range(50):
            loss = self.iterate_over(var_map, start_nodes, l2_node, optimizer)
            losses.append(loss)

        for i, loss in enumerate(losses):
            if i % 10 == 0:
                info("Loss[{}]:{}".format(i, loss))

        final_w = var_map['w']
        final_b = var_map['b']
        debug("W={}".format(final_w))
        debug("b={}".format(final_b))
        final_y_pred = final_w @ x + final_b
        loss = np.sum(np.square(final_y_pred - y_act))
        debug("direct loss calculation:{}".format(loss))
        network_loss = l2_node.value(var_map)
        debug("Network loss:{}".format(network_loss))
        # np.testing.assert_array_almost_equal(final_y_pred, y_act)
        expected_w = np.array([[1.71395966, 2.28604034, 1.42791932],
                               [1.14233545, -0.14233545, 1.2846709]])
        expected_b = np.array([[-1.28604034],
                               [-1.85766455]])
        np.testing.assert_array_almost_equal(final_w, expected_w)
        np.testing.assert_array_almost_equal(final_b, expected_b)
        self.assertTrue( math.fabs(network_loss) < 2.6e-5)

    def iterate_over(self, var_map, start_nodes, l2_node, optimizer):
        for start_node in start_nodes:
            start_node.reset_network_fwd()
        l2_node.reset_network_back()
        for start_node in start_nodes:
            start_node.forward(var_map, None, self)
        loss = l2_node.value(var_map)
        l2_node.backward(1.0, self, var_map, " ")
        l2_node.optimizer_step(optimizer, var_map)
        return loss

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
        optimizer = core.np.Optimization.OptimizerIterator(start_nodes, l2_node)
        log_at_info()
        for _ in range(100):
            loss = optimizer.step(var_map, 1.0)
        debug("Final loss:{}".format(loss))
        self.assertTrue(math.fabs(loss) < 1e-11)


class BasicNetworkSigmoid(BaseComputeNodeTest):
    def test_sigmoid_node(self):
        x_node = node.VarNode('x')
        x = (np.array([[1, -1, 2]])).T
        var_map = {'x': x}
        sigmoid = SigmoidNode(x_node)
        x_node.forward(var_map,None, self)
        value = sigmoid.value(var_map)
        expected_value = 1/(1+np.exp(-x))
        np.testing.assert_array_almost_equal(expected_value, value)
        debug(value)
        sigmoid.backward(np.ones_like(value), self, var_map, " ")
        grad = x_node.grad_value()
        expected_grad = expected_value*(1-expected_value)
        debug(grad)
        np.testing.assert_array_almost_equal(expected_grad/expected_grad.size, grad)


if __name__ == '__main__':
    unittest.main()
