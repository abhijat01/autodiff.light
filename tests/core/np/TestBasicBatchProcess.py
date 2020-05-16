import unittest
import numpy as np

import core.np.Nodes as node
from . import BaseComputeNodeTest
from core import log_at_info, info
from core.np.Loss import L2DistanceSquaredNorm


class UnitNetworkBatch(BaseComputeNodeTest):
    def test_linear_transformation(self):
        np.random.seed(100)
        w_node = node.VarNode('w')
        x_node = node.VarNode('x')
        ya_node = node.VarNode('y_a')

        w = np.array([[1, 2, 1], [2, 0, -1]])
        x = np.array([[0.54340494, 0.27836939, 0.42451759, 0.84477613, 0.00471886],
                      [0.12156912, 0.67074908, 0.82585276, 0.13670659, 0.57509333],
                      [0.89132195, 0.20920212, 0.18532822, 0.10837689, 0.21969749]])
        y_act = np.array([[0.97862378, 0.81168315, 0.17194101, 0.81622475, 0.27407375],
                          [0.43170418, 0.94002982, 0.81764938, 0.33611195, 0.17541045]])
        info("Printing x...")
        info(x)
        info("printing y_pred")
        info(y_act)

        var_map = {'w': w, 'x': x, 'y_a': y_act}

        wx_node = node.MatrixMultiplication(w_node, x_node)
        l2_node = L2DistanceSquaredNorm(wx_node, ya_node)
        log_at_info()
        w_node.forward(var_map)
        x_node.forward(var_map)
        ya_node.forward(var_map)
        l2_node.backward(1.0, self, var_map, " ")
        info(wx_node.value(var_map))
        info("grad...")
        info(wx_node.grad_value())


if __name__ == '__main__':
    unittest.main()
