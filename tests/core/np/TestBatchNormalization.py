import unittest
import tests.core.np.BaseTests as base
import numpy as np
import core.np.Nodes as node
import core.np.normalization as norm
from core import debug, info

class BatchNormalizationTest(base.BaseComputeNodeTest):

    def test_something(self):
        x = np.array([[1, 2, 3], [3, 4, 1]])
        x_node = node.VarNode('x')
        bnorm = norm.BatchNormalization(x_node)
        x_node.forward({'x': x})
        norm_value = bnorm.value()
        debug("normalized value = np.{}".format(repr(norm_value)))
        bnorm.backward(np.ones_like(norm_value), self, {'x':x})
        grad_at_x = x_node.grad_value()
        debug("grad_at_x = np.{}".format(repr(grad_at_x)))
        debug("[BatchNormalizationTest.test_something()] bnorm.gamma_grad = np.{}".format(repr(bnorm.gamma_grad)))
        debug("[BatchNormalizationTest.test_something()] bnorm.beta_grad = np.{}".format(repr(bnorm.beta_grad)))


if __name__ == '__main__':
    unittest.main()
