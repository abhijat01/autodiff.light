from .BaseTests import BaseComputeNodeTest
from core.np.Activations import SigmoidNode, RelUNode, Softmax
from core.np.Loss import L2DistanceSquaredNorm
import core.np.Nodes as node
import numpy as np
from core import debug


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
        ones = np.ones_like(value)
        relu.backward(ones, self, var_map, " ")
        grad = x_node.grad_value()
        np.testing.assert_almost_equal(expected_grad, grad)

    def test_softmax(self):
        x = np.array([[1, 3, -1, 0], [0, 9, 1, 3]]).T
        x_node = node.VarNode('x')
        var_map = {'x': x}
        softmax = Softmax(x_node)
        x_node.forward(var_map, None, self)
        expected_value = np.array([[1.12457214e-01, 1.23048334e-04],
                                   [8.30952661e-01, 9.97070980e-01],
                                   [1.52194289e-02, 3.34480051e-04],
                                   [4.13706969e-02, 2.47149186e-03]])
        value = softmax.value(var_map)
        np.testing.assert_almost_equal(expected_value, value)
        ones = np.ones_like(value)
        softmax.backward(ones, self, var_map, " ")
        grad = x_node.grad_value()
        debug("grad = np.{}".format(repr(grad)))

    def test_softmax_grad(self):
        r"""
        See TestActivations.Softmax.ipynb for corresponding pytorch calculations
        :return:
        """
        x = np.array([[1, 3, -1, 0], [0, 9, 1, 3]]).T
        x_node = node.VarNode('x')

        softmax = Softmax(x_node)
        target = np.zeros(x.shape)
        target_node = node.VarNode('target')
        var_map = {'x': x, 'target': target}
        l2loss = L2DistanceSquaredNorm(softmax, target_node)
        x_node.forward(var_map, None, self)
        target_node.forward(var_map, None, self)

        expected_value = np.array([[1.12457214e-01, 1.23048334e-04],
                                   [8.30952661e-01, 9.97070980e-01],
                                   [1.52194289e-02, 3.34480051e-04],
                                   [4.13706969e-02, 2.47149186e-03]])
        value = softmax.value(var_map)
        np.testing.assert_almost_equal(expected_value, value)
        loss_value = l2loss.value(var_map)
        debug("Loss = {}".format(loss_value))
        init_grad = np.ones(x.shape)
        l2loss.backward(init_grad, self, var_map, " ")
        x_grad = x_node.grad_value()

        expected_grad = np.array([[-0.01666096, -0.00003058],
                                  [0.02615019, 0.00072642],
                                  [-0.00262479, -0.0000831],
                                  [-0.00686445, -0.00061274]])

        debug("x_grad = np.{}".format(repr(x_grad)))
        np.testing.assert_almost_equal(expected_grad, x_grad)
