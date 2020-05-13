from .BaseTests import BaseComputeNodeTest

from core.np.Loss import CrossEntropy
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
        predicted_node.forward(var_map, None, self)
        target_node.forward(var_map, None, self)
        loss = cross_entropy.value(var_map)
        debug("loss = {}".format(loss))
        expected_loss = 6.188115770824936
        self.assertAlmostEqual(expected_loss, loss)
        cross_entropy.backward(1.0, self,var_map, " ")
        x_grad = predicted_node.grad_value()
        debug("x_grad = np.{}".format(repr(x_grad)))
