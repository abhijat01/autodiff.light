import unittest
from . import BaseComputeNodeTest
import numpy as np
import core.np.Nodes as node
import core.np.Convolution as conv
from core import debug, info, log_at_info


class MaxPool2DUnitTests(BaseComputeNodeTest):

    def setUp(self):
        x = np.array([[1, 2, -1, 4], [2, -1, 3, 1], [4, 9, -4, 5]])
        debug("x = np.{}".format(repr(x)))
        self.x_node = node.VarNode('x')
        self.var_map = {'x': x}
        self.max_pool_node = conv.MaxPool2D(self.x_node, pool_size=(2, 2), name="maxpool")
        debug("x = np.{}".format(repr(x)))

    def forward(self):
        self.x_node.forward(self.var_map)

    def test_forward(self):
        self.forward()
        value = self.max_pool_node.value()
        info("maxpool = np.{}".format(repr(value)))
        expected_value = np.array([[2, 3, 4],
                                   [9, 9, 5]])
        np.testing.assert_almost_equal(expected_value, value)

    def test_backprop(self):
        self.forward()
        value = self.max_pool_node.value()
        debug("value = np.{}".format(repr(value)))
        ones = np.ones_like(value)
        self.max_pool_node.backward(ones, self, self.var_map)
        grad_from_maxpool = self.x_node.total_incoming_gradient()
        debug("grad_from_maxpool = np.{}".format(repr(grad_from_maxpool)))
        expected_grad = np.array([[0, 1, 0, 1],
                                  [0, 0, 1, 0],
                                  [0, 2, 0, 1]])
        np.testing.assert_almost_equal(expected_grad, grad_from_maxpool)


if __name__ == '__main__':
    unittest.main()
