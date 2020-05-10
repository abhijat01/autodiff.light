import unittest
import numpy as np
import core.np.Nodes as node
from core import debug, info, log_at_info
from . import LinearModel


class DenseLayerStandAlone(unittest.TestCase):
    def test_basic_op(self):
        np.random.seed(100)

        x = np.array([[1, -1], [2, 3], [-1, -2]], dtype=np.float)
        y = np.array([[-1, 1], [-3, -1]], dtype=np.float)
        model_w = np.array([[1, 3, -1], [0, -4, 2]], dtype=np.float)
        model_b = np.array([[-3, 2]], dtype=np.float).reshape((model_w.shape[0], 1))

        x_node = node.VarNode('x')
        dense = node.DenseLayer(x_node, 2, model_w, model_b)
        ypred_node = node.VarNode('y_pred')
        l2_node = node.L2DistanceSquaredNorm(dense, ypred_node)

        var_map = {'x': x, 'y_pred': y}
        x_node.forward(var_map, None, self)
        ypred_node.forward(var_map, None, self)

        log_at_info()
        value = dense.value(var_map)
        info("Dense node value = np.{}".format(repr(value)))
        value = l2_node.value(var_map)
        info("L2 node value:{}".format(value))

        info("--")
        info("Printing weights (not updated yet)")
        info("------------------------------------------")
        info("Linear layer weight:{}".format(repr(dense.get_w())))
        info("Linear layer bias:{}".format(repr(dense.get_b())))

        def optimizer_function(_w, grad):
            return _w - 0.001 * grad

        optimizer = node.OptimizerIterator([x_node, ypred_node], l2_node, optimizer_function)
        optimizer.step(var_map, 1.0)
        info("Printing after updating weights")
        info("------------------------------------------")
        info("Linear layer weight:{}".format(repr(dense.get_w())))
        info("Linear layer bias:{}".format(repr(dense.get_b())))
        info("w_grad = np.{}".format(repr(dense.get_w_grad())))
        info("b_grad = np.{}".format(repr(dense.get_b_grad())))
        expected_weight = np.array([[1., 2.985, -0.991],
                                    [-0.004, -3.9755, 1.9845]])
        expected_bias = np.array([[-3.006],
                                  [2.009]])
        expected_w_grad = np.array([[0., 15., -9.],
                                    [4., -24.5, 15.5]])
        expected_b_grad = np.array([[6.],
                                    [-9.]])

        np.testing.assert_almost_equal(expected_weight, dense.get_w())
        np.testing.assert_almost_equal(expected_w_grad, dense.get_w_grad())
        np.testing.assert_almost_equal(expected_bias, dense.get_b())
        np.testing.assert_almost_equal(expected_b_grad, dense.get_b_grad())

    def test_linear_optimization(self):
        np.random.seed(100)
        x_node = node.VarNode('x')
        net_w = np.array([[-1, -3, 1], [0, 4, -2]])
        net_b = np.array([3, -2]).reshape((2,1))
        dense = node.DenseLayer(x_node, 2, net_w, net_b)
        y_node = node.VarNode('y')
        l2_node = node.L2DistanceSquaredNorm(dense, y_node)
        model_w = np.array([[1, 3, -1], [0, -4, 2]])
        model_b = np.array([[-3, 2]]).reshape((model_w.shape[0], 1))
        model = LinearModel(model_w, model_b)

        learning_rate = 0.01

        def optimizer_function(_w, grad):
            return _w - learning_rate * grad

        optimizer = node.OptimizerIterator([x_node, y_node], l2_node, optimizer_function)
        log_at_info()
        epoch = 0
        for x, y in model.data(50000, 8):
            var_map = {'x': x, 'y': y}
            loss = optimizer.step(var_map, 1.0)
            if epoch % 1000 == 0:
                debug("[{}] Loss:{}".format(epoch, loss))
                if epoch % 1000 == 0:
                    info("[{}] Loss:{}".format(epoch, loss))
            epoch += 1

        dense_w = dense.get_w()
        dense_b = dense.get_b()
        info("w = np.{}".format(repr(dense_w)))
        info("b = np.{}".format(repr(dense_b)))
        np.testing.assert_array_almost_equal(dense_w, model_w, 3)
        np.testing.assert_array_almost_equal(dense_b, model_b, 3)

    def simple_name(self):
        return self.__class__.__name__


if __name__ == '__main__':
    unittest.main()
