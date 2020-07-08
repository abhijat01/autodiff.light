from .BaseTests import BaseComputeNodeTest

from core.np.Loss import CrossEntropy, SoftmaxCrossEntropy
from core.np.Activations import Softmax
import core.np.Nodes as node
import numpy as np
from core import debug


class SimpleCrossEntryTestCase(BaseComputeNodeTest):
    def test_cross_entropy(self):
        predicted = np.array([[1, 3, -1, 0], [0, 9, 1, 3.]]).T
        target = np.array([[0, 1, 0, 0], [0, 0, 0, 1]]).T
        predicted_node = node.VarNode('predicted')
        target_node = node.VarNode('target')
        softmax = Softmax(predicted_node)
        cross_entropy = CrossEntropy(softmax, target_node)
        var_map = {'predicted': predicted, 'target': target}
        predicted_node.forward(var_map)
        target_node.forward(var_map)
        loss = cross_entropy.value()
        debug("loss = {}".format(loss))
        expected_loss = 6.188115770824936
        self.assertAlmostEqual(expected_loss, loss)
        cross_entropy.backward(1.0, self, var_map)
        x_grad = predicted_node.total_incoming_gradient()
        debug("x_grad = np.{}".format(repr(x_grad)))
        # Note that this grad is  1/8 the size reported by pytorch
        # because pytorch does not average out during softmax for CrossEntropy
        # whereas I use softmax node
        expected_grad = np.array([[1.40571517e-02, 1.53810418e-05],
                                  [-2.11309174e-02, 1.24633872e-01],
                                  [1.90242861e-03, 4.18100064e-05],
                                  [5.17133712e-03, -1.24691064e-01]])
        np.testing.assert_array_almost_equal(expected_grad, x_grad)

    def test_softmax_cross_entropy(self):
        predicted = np.array([[1, 3, -1, 0], [0, 9, 1, 3.]]).T.reshape(4, 2)
        debug("[SoftmaxTestCase.test_batch()] predicted = np.{}".format(repr(predicted)))
        target = np.array([[0, 1, 0, 0], [0, 0, 0, 1]]).T.reshape(4, 2)
        ctx = node.ComputeContext({'pred': predicted, 'target': target})
        pred_node = node.VarNode('pred')
        target_node = node.VarNode('target')
        sx = SoftmaxCrossEntropy(pred_node, target_node)
        pred_node.forward(ctx)
        target_node.forward(ctx)
        loss = sx.value()
        debug("[SimpleCrossEntryTestCase.test_softmax_cross_entropy()] loss = {}".format(repr(loss)))
        np.testing.assert_equal(6.188115770824936, loss)
        sx.backward(1.0, self, ctx)
        grad_at_p = pred_node.total_incoming_gradient()
        debug("[SimpleCrossEntryTestCase.test_softmax_cross_entropy()] grad_at_p = np.{}".format(repr(grad_at_p)))
        expected_grad = np.array([[1.12457214e-01, 1.23048334e-04],
                                  [-1.69047339e-01, 9.97070980e-01],
                                  [1.52194289e-02, 3.34480051e-04],
                                  [4.13706969e-02, -9.97528508e-01]])
        np.testing.assert_array_almost_equal(expected_grad, grad_at_p)
