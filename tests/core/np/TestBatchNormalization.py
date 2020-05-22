import unittest
import tests.core.np.BaseTests as base
import numpy as np
import core.np.Nodes as node
import core.np.Specialized as sp
from core import  debug, info

class BatchNormalizationTest(base.BaseComputeNodeTest):

    def test_something(self):
        x = np.array([[1, 2, 3], [3, 4, 1]])
        x_node = node.VarNode('x')
        bnorm = sp.BatchNormalization(x_node)
        x_node.forward({'x':x})
        norm_value = bnorm.value()
        debug(" [BatchNormalizationTest.test_something()] norm_value = np.{}".format(repr(norm_value)))
        bnorm.backward(np.ones_like(norm_value), self, {'x':x})
        grad_at_x = x_node.grad_value()
        debug(" [BatchNormalizationTest.test_something()] grad_at_x = np.{}".format(repr(grad_at_x)))


if __name__ == '__main__':
    unittest.main()
