import unittest
import numpy as np

from core.np.Loss import L2DistanceSquaredNorm
import core.np.Nodes as node
from . import BaseComputeNodeTest


class BinaryOpBackProp(BaseComputeNodeTest):

    def test_matrix_sum(self):
        a_node = node.VarNode('a')
        b_node = node.VarNode('b')
        a = np.array([[1, 2, 3], [-1, -3, 0]])
        b = np.array([[0, 1, -1], [2, 0, 1]])
        sum_node = node.MatrixAddition(a_node, b_node)
        var_map = {'a': a, 'b': b}
        a_node.forward(var_map, None, self)
        b_node.forward(var_map, None, self)
        matrix_sum = sum_node.value(var_map)
        expected_sum = a + b
        print(matrix_sum)
        np.testing.assert_array_almost_equal(expected_sum, matrix_sum)
        start_grad = np.ones_like(a)
        sum_node.backward(start_grad, self, var_map)
        grad_at_a = a_node.grad_value()
        grad_at_b = b_node.grad_value()
        print(grad_at_a)
        print("-------------")
        print(grad_at_b)
        np.testing.assert_array_almost_equal(grad_at_a, start_grad)
        np.testing.assert_array_almost_equal(grad_at_b, start_grad)

    def test_l2norm(self):
        y_pred = np.array([[1, 2, 3]]).T
        y_act = np.array([[1, 1, 1]]).T
        y_del = y_pred - y_act
        expected_norm = np.sum(np.square(y_del))/y_del.size

        y_p_node = node.VarNode('y_p')
        y_a_node = node.VarNode('y_a')
        var_map = {'y_p': y_pred, 'y_a': y_act}
        l2norm_node = L2DistanceSquaredNorm(y_p_node, y_a_node)

        y_p_node.forward(var_map, None, self)
        y_a_node.forward(var_map, None, self)
        l2norm = l2norm_node.value(var_map)
        print(l2norm)
        self.assertEqual(l2norm, expected_norm)

        ones = np.ones_like(y_pred)
        l2norm_node.backward(ones, self, var_map, " ")
        grad_at_yp = y_p_node.grad_value()
        print("start print grad at y_p:")
        print(grad_at_yp)
        print("end print grad at y_p")

    def test_matrix_prd(self):
        w_node = node.VarNode('w')
        x_node = node.VarNode('x')
        w = np.array([[1, 3, 0], [0, 1, -1]])
        x = (np.array([[1, -1, 2]])).T
        w_grad_expected = np.array([x[:, 0], x[:, 0]])
        local_grad = np.array([[1, 1]]).T
        x_grad_expected = np.multiply(w, local_grad).sum(axis=0).T
        x_grad_expected = np.reshape(x_grad_expected, (len(x_grad_expected), 1))

        mult_node = node.MatrixMultiplication(w_node, x_node, name="wx")
        var_map = {'x': x, 'w': w}
        x_node.forward(var_map, None, self)
        self.assertIsNone(mult_node.value(var_map))
        w_node.forward(var_map, None, self)
        value = mult_node.value(var_map)
        expected = w @ x
        np.testing.assert_array_almost_equal(expected, value)
        mult_node.backward(local_grad, self, var_map, " ")
        w_grad = w_node.grad_value()
        print("---- printing w_grad ---")
        print(w_grad)
        np.testing.assert_array_almost_equal(w_grad, w_grad_expected)
        print("---- end printing   ----")
        x_grad = x_node.grad_value()
        print("---- printing x_grad ---")
        print(x_grad)
        np.testing.assert_array_almost_equal(x_grad_expected, x_grad)
        print("---- end printing   ----")


if __name__ == '__main__':
    unittest.main()
