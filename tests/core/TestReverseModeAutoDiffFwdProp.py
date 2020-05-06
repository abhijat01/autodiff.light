import unittest
import math
import core.ReverseModeAutoDiff as rev


class BinaryOpFwdPropTestCases(unittest.TestCase):
    def test_with_constants(self):
        c1 = rev.ConstantOpNode(4)
        c2 = rev.ConstantOpNode(5)
        self.assertEqual(c1.value({}), 4)
        self.assertEqual(c2.value({}), 5)

        bin_node = rev.SumNode(c1, c2)
        self.assertIsNone(bin_node.value({}))
        bin_node.forward(None, {})
        self.assertIsNone(bin_node.value({}))
        bin_node.forward(None, {})
        self.assertEqual(bin_node.value({}), 9)

        bin_node = rev.ProdNode(c1, c2)
        self.assertIsNone(bin_node.value({}))
        bin_node.forward(None, {})
        self.assertIsNone(bin_node.value({}))
        bin_node.forward(None, {})
        self.assertEqual(bin_node.value({}), 20)

    def test_with_vars(self):
        var_dict = {'x': 2, 'y': 3}
        x = rev.VarNode('x')
        y = rev.VarNode('y')

        self.assertEqual(2, x.value(var_dict))
        self.assertEqual(3, y.value(var_dict))

        bin_node = rev.SumNode(x, y)
        self.assertIsNone(bin_node.value(var_dict))
        x.forward(None, var_dict)
        self.assertIsNone(bin_node.value(var_dict))
        y.forward(None, var_dict)
        self.assertEqual(bin_node.value(var_dict), 5)

        bin_node = rev.ProdNode(x, y)
        self.assertIsNone(bin_node.value(var_dict))
        x.forward(None, var_dict)
        self.assertIsNone(bin_node.value(var_dict))
        y.forward(None, var_dict)
        self.assertEqual(bin_node.value(var_dict), 6)

    def test_mixed(self):
        var_dict = {'x': 2, 'y': 3}
        x = rev.ConstantOpNode(10)
        y = rev.VarNode('y')

        self.assertEqual(10, x.value(var_dict))
        self.assertEqual(3, y.value(var_dict))

        bin_node = rev.SumNode(x, y)
        self.assertIsNone(bin_node.value(var_dict))
        x.forward(None, var_dict)
        self.assertIsNone(bin_node.value(var_dict))
        y.forward(None, var_dict)
        self.assertEqual(bin_node.value(var_dict), 13)

        bin_node = rev.ProdNode(x, y)
        self.assertIsNone(bin_node.value(var_dict))
        x.forward(None, var_dict)
        self.assertIsNone(bin_node.value(var_dict))
        y.forward(None, var_dict)
        self.assertEqual(bin_node.value(var_dict), 30)


class BaydinPaperFwdPropComputeTest(unittest.TestCase):
    r"""
    Test case from
    Automatic Differentiation in Machine Learning: a Survey, Baydin et al., 2018
    """

    def test_expression(self):
        x1 = rev.VarNode('x1')
        x2 = rev.VarNode('x2')
        log_node = rev.LogNode(x1)
        prod_node = rev.ProdNode(x1, x2)
        sum_node = rev.SumNode(log_node, prod_node)
        sine_node = rev.SinNode(x2)
        diff_node = rev.DiffNode(sum_node, sine_node)
        var_dict = {'x1':2, 'x2':5}
        x1.forward(None, var_dict)
        x2.forward(None, var_dict)
        self.assertIsNotNone(diff_node.value(var_dict))
        actual = math.log(2)+10-math.sin(5)
        self.assertAlmostEqual(diff_node.value(var_dict), actual)


if __name__ == '__main__':
    unittest.main()
