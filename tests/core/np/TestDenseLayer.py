import unittest
import numpy as np
import core.np.Nodes as node
from core import debug, info, log_at_info
from . import LinearModel


class DenseLayerStandAlone(unittest.TestCase):
    def test_basic_op(self):
        np.random.seed(100)
        x_node = node.VarNode('x')
        dense = node.DenseLayer(x_node, 2)
        ypred_node = node.VarNode('y_pred')
        l2_node = node.L2DistanceSquaredNorm(dense, ypred_node)

        x = np.random.rand(3, 2)
        y = np.random.rand(2, 2)
        debug("x = np.{}".format(repr(x)))
        var_map = {'x': x, 'y_pred': y}

        x_node.forward(var_map, None, self)
        ypred_node.forward(var_map, None, self)

        value = dense.value(var_map)
        debug("dense node value = np.{}".format(value))
        value = l2_node.value(var_map)
        debug("L2 node value:{}".format(value))

        start_grad = 1.
        l2_node.backward(start_grad, self, var_map, " ")
        weight_grads = dense.get_component_grads()
        debug("w_grad = np.{}".format(repr(weight_grads['w'])))
        debug("b_grad = np.{}".format(repr(weight_grads['b'])))

    def test_linear_optimization(self):
        np.random.seed(100)
        x_node = node.VarNode('x')
        dense = node.DenseLayer(x_node, 2)
        y_node = node.VarNode('y')
        l2_node = node.L2DistanceSquaredNorm(dense, y_node)
        model_w = np.array([[1, 3, -1], [0, -4, 2]])
        model_b = np.array([[-3, 2]]).reshape((model_w.shape[0], 1))
        model = LinearModel(model_w, model_b)

        def optimizer_function(_w, grad):
            return _w - 0.001 * grad

        optimizer = node.OptimizerIterator([x_node, y_node], l2_node, optimizer_function)
        log_at_info()
        epoch = 0
        for x, y in model.data(40000, 400):
            var_map = {'x': x, 'y': y}
            loss = optimizer.step(var_map, 1.0)
            if epoch % 1000 == 0:
                debug("[{}] Loss:{}".format(epoch, loss))
                if epoch % 10000 == 0:
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
