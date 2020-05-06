import unittest
import numpy as np
import core.np.Nodes as node


class BinaryOpFwdProp(unittest.TestCase):
    def test_basic(self):
        w_node = node.VarNode('w')
        x_node = node.VarNode('x')
        w = np.array([[1, 3, 0], [0, 1, -1]])
        x = np.array([[1, -1], [0, 2], [9, 1]])
        mult_node = node.MatrixMult(w_node, x_node)
        var_map = {'x': x, 'w': w}
        x_node.forward(var_map, None, self)
        self.assertIsNone(mult_node.value(var_map))
        w_node.forward(var_map, None, self)
        value = mult_node.value(var_map)
        expected = w @ x
        np.testing.assert_array_almost_equal(expected, value)
        print(value)
        self.assertIsNotNone(x_node.value(var_map))
        mult_node.reset_network_fwd()
        # Just checking
        # Not none because fwd should start from start vars
        self.assertIsNotNone(x_node.value(var_map))
        b_node = node.VarNode('b')
        b = np.array([-1, -1])
        var_map['b'] = b
        sum_node = node.MatrixAddition(mult_node, b_node)

        var_nodes = [x_node, w_node, b_node]
        for var_node in var_nodes:
            var_node.forward(var_map, None, self)

        expected = expected+b
        np.testing.assert_array_almost_equal(expected, sum_node.value(var_map))
        print(sum_node.value(var_map))


if __name__ == '__main__':
    unittest.main()
