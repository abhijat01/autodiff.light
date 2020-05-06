import unittest
import math
import core.ReverseModeAutoDiff as rev


class BinaryOpBackPropTestCases(unittest.TestCase):

    def get_simple_name(self):
        return BinaryOpBackPropTestCases.__name__

    def test_with_vars(self):
        var_dict = {'x': 2, 'y': 3}
        x = rev.VarNode('x')
        y = rev.VarNode('y')

        self.assertEqual(2, x.value(var_dict))
        self.assertEqual(3, y.value(var_dict))

        x_end = rev.EndValueCollectorNode(x)
        y_end = rev.EndValueCollectorNode(y)
        x.forward(None, var_dict)
        y.forward(None, var_dict)
        x_end.backward(None, self, var_dict)
        y_end.backward(None, self, var_dict)

        self.assertEqual(1, x.grad_value(var_dict))

        x = rev.VarNode('x')
        y = rev.VarNode('y')
        bin_node = rev.SumNode(x, y)
        self.assertIsNone(bin_node.value(var_dict))
        x.forward(None, var_dict)
        self.assertIsNone(bin_node.value(var_dict))
        y.forward(None, var_dict)
        self.assertEqual(bin_node.value(var_dict), 5)

        bin_node.backward(1, self, var_dict)
        self.assertEqual(1, x.grad_value(var_dict))
        self.assertEqual(1, y.grad_value(var_dict))

        x = rev.VarNode('x')
        y = rev.VarNode('y')
        bin_node = rev.ProdNode(x, y)
        x.reset_back()
        y.reset_back()
        self.assertIsNone(bin_node.value(var_dict))
        x.forward(None, var_dict)
        self.assertIsNone(bin_node.value(var_dict))
        y.forward(None, var_dict)
        self.assertEqual(bin_node.value(var_dict), 6)
        bin_node.backward(1, self, var_dict)
        self.assertEqual(var_dict['y'], x.grad_value(var_dict))
        self.assertEqual(var_dict['x'], y.grad_value(var_dict))


class BaydinPaperBackPropComputeTest(unittest.TestCase):
    r"""
    Test case from:
    Automatic Differentiation in Machine Learning: a Survey, Baydin et al., 2018
    """

    def test_expression(self):
        x1 = rev.VarNode('x1')
        x2 = rev.VarNode('x2')
        log_node = rev.LogNode(x1, "v1|ln")
        prod_node = rev.ProdNode(x1, x2, "v2|prod")
        sum_node = rev.SumNode(log_node, prod_node, "v4|sum")
        sine_node = rev.SinNode(x2, "v3|sin")
        diff_node = rev.DiffNode(sum_node, sine_node, "v5|diff")
        var_dict = {'x1': 2, 'x2': 5}
        x1.forward(None, var_dict)
        x2.forward(None, var_dict)
        self.assertIsNotNone(diff_node.value(var_dict))
        actual = math.log(2) + 10 - math.sin(5)
        self.assertAlmostEqual(diff_node.value(var_dict), actual)

        diff_node.backward(1.0, self, var_dict, " ")

        print("dx1={}".format(x1.grad_value(var_dict)))
        print("dx2={}".format(x2.grad_value(var_dict)))
        self.assertAlmostEqual(5.5, x1.grad_value(var_dict))
        self.assertAlmostEqual(1.71633781453677, x2.grad_value(var_dict))

    def get_simple_name(self):
        return "f"


if __name__ == '__main__':
    unittest.main()
