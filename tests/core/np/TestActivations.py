from .BaseTests import BaseComputeNodeTest
from core.np.Activations import SigmoidNode, RelUNode
import core.np.Nodes as node
import numpy as np


class SimpleActivationTests(BaseComputeNodeTest):
    def test_sigmoid(self):
        x = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [-1, 0, 1, 3]])
        x_node = node.VarNode('x')
        var_map = {'x': x}
        sigmoid = SigmoidNode(x_node)
        x_node.forward(var_map, None, self)
        value = sigmoid.value(var_map)
        expected_value = np.array([[0.73105858, 0.88079708, 0.95257413, 0.98201379],
                                   [0.95257413, 0.98201379, 0.99330715, 0.99752738],
                                   [0.26894142, 0.5, 0.73105858, 0.95257413]])
        np.testing.assert_almost_equal(expected_value, value)
        initial_grad = np.ones_like(value)
        sigmoid.backward(initial_grad, self, var_map, " ")
        grad_from_sigmoid = x_node.grad_value()
        expected_grad = np.array([[0.19661193, 0.10499359, 0.04517666, 0.01766271],
                                  [0.04517666, 0.01766271, 0.00664806, 0.00246651],
                                  [0.19661193, 0.25, 0.19661193, 0.04517666]])
        np.testing.assert_almost_equal(expected_grad, grad_from_sigmoid)

    def test_relu(self):
        x = np.array([[1, -2, 3, -4], [2, 0, -9, .5]])
        x_node = node.VarNode('x')
        var_map = {'x': x}
        relu = RelUNode(x_node)
        x_node.forward(var_map, None, self)
        expected_value = np.array([[1, 0, 3, 0], [2, 0, 0, .5]])
        value = relu.value(var_map)
        np.testing.assert_almost_equal(expected_value, value)
        expected_grad = np.array([[1, 0, 1, 0], [1, 0, 0, 1]])
        ones = np.ones_like(var_map)
        relu.backward(ones, self, var_map, " ")
        grad = x_node.grad_value()
        np.testing.assert_almost_equal(expected_grad, grad)
