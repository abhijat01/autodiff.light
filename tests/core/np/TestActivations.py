from .BaseTests import BaseComputeNodeTest
from core.np.Activations import SigmoidNode, RelUNode, Softmax
from core.np.Loss import L2DistanceSquaredNorm, LogitsCrossEntropy
import core.np.Nodes as node
import numpy as np
from core import debug, info, log_at_info
from core.np.utils import to_one_hot


class SimpleActivationTests(BaseComputeNodeTest):
    r"""
    Over time, we should check these tests in pytorch (preferred) or tensorflow/keras
    """
    def test_sigmoid(self):
        r"""
        See TestActivations.Sigmoid.ipynb for the corresponding pytorch calculations
        :return:
        """
        x = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [-1, 0, 1, 3]])
        x_node = node.VarNode('x')
        target = np.zeros(x.shape)
        target_node = node.VarNode('target')
        var_map = {'x': x, 'target': target}
        sigmoid = SigmoidNode(x_node)
        l2loss = L2DistanceSquaredNorm(sigmoid, target_node)
        x_node.forward(var_map)
        target_node.forward(var_map)
        value = sigmoid.value()
        expected_value = np.array([[0.73105858, 0.88079708, 0.95257413, 0.98201379],
                                   [0.95257413, 0.98201379, 0.99330715, 0.99752738],
                                   [0.26894142, 0.5, 0.73105858, 0.95257413]])
        np.testing.assert_almost_equal(expected_value, value)
        loss = l2loss.value()
        info("L2 Loss:{}".format(loss))
        log_at_info()
        l2loss.backward(1.0, self, var_map)
        x_grad = x_node.grad_value()
        expected_x_grad = np.array([[0.28746968, 0.18495609, 0.08606823, 0.03469004],
                                    [0.08606823, 0.03469004, 0.01320712, 0.00492082],
                                    [0.10575419, 0.25, 0.28746968, 0.08606823]])
        info("-------------------------------------------------------------")
        info("x_grad = np.{}".format(repr(x_grad)))
        info("x_grad_expected= np.{}".format(repr(expected_x_grad)))
        np.testing.assert_almost_equal(expected_x_grad, x_grad)

    def test_relu(self):
        x = np.array([[1, -2, 3, -4], [2, 0, -9, .5]])
        x_node = node.VarNode('x')
        var_map = {'x': x}
        relu = RelUNode(x_node)
        x_node.forward(var_map)
        expected_value = np.array([[1, 0, 3, 0], [2, 0, 0, .5]])
        value = relu.value()
        np.testing.assert_almost_equal(expected_value, value)

        ones = np.ones_like(value)
        relu.backward(ones, self, var_map)
        grad = x_node.grad_value()
        expected_grad = np.array([[1, 0, 1, 0], [1, 0, 0, 1]])
        np.testing.assert_almost_equal(expected_grad, grad)

    def test_softmax(self):
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
        x_node.forward(var_map)
        target_node.forward(var_map)

        expected_value = np.array([[1.12457214e-01, 1.23048334e-04],
                                   [8.30952661e-01, 9.97070980e-01],
                                   [1.52194289e-02, 3.34480051e-04],
                                   [4.13706969e-02, 2.47149186e-03]])
        value = softmax.value()
        np.testing.assert_almost_equal(expected_value, value)
        loss_value = l2loss.value()
        debug("Loss = {}".format(loss_value))
        l2loss.backward(1.0, self, var_map)
        x_grad = x_node.grad_value()

        expected_grad = np.array([[-0.01666096, -0.00003058],
                                  [0.02615019, 0.00072642],
                                  [-0.00262479, -0.0000831],
                                  [-0.00686445, -0.00061274]])

        debug("x_grad = np.{}".format(repr(x_grad)))
        np.testing.assert_almost_equal(expected_grad, x_grad)

    def test_logit_cross_entropy(self):
        logits = np.array([[2, 1, 4, -1], [3, 2, 1, -9]])
        target_values = np.array([2, 0])
        one_hot_target = to_one_hot(target_values, logits.shape[1] - 1)
        debug(" [SimpleActivationTests.test_logit_cross_entropy()] one_hot_target = np.{}".format(repr(one_hot_target)))
        pred_node = node.VarNode('yp')
        target_node = node.VarNode('yt')
        var_map = {'yp': logits.T, 'yt': one_hot_target}
        lx = LogitsCrossEntropy(pred_node, target_node)
        pred_node.forward(var_map)
        target_node.forward(var_map)
        value = lx.value()
        expected = 0.2915627072172198
        debug(" [SimpleActivationTests.test_logit_cross_entropy()] value = {}".format(repr(value)))
        self.assertAlmostEqual(expected, value)
        lx.backward(1.0, self, var_map)
        grad = pred_node.grad_value()
        debug(" [LogitCrossEntropyTests.test_logit_cross_entropy()] grad = np.{}".format(repr(grad)))
        expected_grad = np.array([[5.67748097e-02, -1.67380882e-01],
                                  [2.08862853e-02, 1.22363735e-01],
                                  [-8.04877463e-02, 4.50151026e-02],
                                  [2.82665133e-03, 2.04368250e-06]])
        debug(" [LogitCrossEntropyTests.test_logit_cross_entropy()] expected_grad = np.{}".format(repr(expected_grad)))
        np.testing.assert_array_almost_equal(expected_grad, grad)
