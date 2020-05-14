import unittest
import numpy as np

import core.np.Optimization
from core.np.Loss import L2DistanceSquaredNorm
import core.np.Nodes as node
from core import debug, info, log_at_info
from . import LinearModel
from . import BaseComputeNodeTest
import matplotlib.pyplot as plt


class DenseLayerStandAlone(BaseComputeNodeTest):
    def setUp(self):
        self.model_w = np.array([[1, 3, -1], [0, -4, 2]], dtype=np.float)
        self.model_b = np.array([[-3, 2]], dtype=np.float).reshape((self.model_w.shape[0], 1))
        self.model = LinearModel(self.model_w, self.model_b)

    def test_basic_op(self):
        np.random.seed(100)

        x = np.array([[1, -1], [2, 3], [-1, -2]], dtype=np.float)
        y = np.array([[-1, 1], [-3, -1]], dtype=np.float)

        x_node = node.VarNode('x')
        ypred_node = node.VarNode('y_pred')

        dense = node.DenseLayer(x_node, 2, self.model_w, self.model_b)
        l2_node = L2DistanceSquaredNorm(dense, ypred_node)

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

        optim_func = self.rate_adjustable_optimizer_func(0.001)

        optimizer = core.np.Optimization.OptimizerIterator([x_node, ypred_node], l2_node, optim_func)
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

    @unittest.skip("This should be moved to notebook")
    def test_compare_optimizations(self):
        opt1 = self.rate_adjustable_optimizer_func(0.01)
        losses_gd = self.do_linear_optimization(opt1, epochs=20000, do_assert=True)

        opt2 = core.np.Optimization.AdamOptimizer(lr=0.01)
        lossed_adam = self.do_linear_optimization(opt2, epochs=20000, do_assert=True)

        start_idx = 100
        plt.plot(losses_gd[start_idx:,0], losses_gd[start_idx:,1], 'r')
        plt.plot(lossed_adam[start_idx:,0], lossed_adam[start_idx:,1], 'g')
        plt.show()

    def test_linear_optimization(self):
        opt2 = core.np.Optimization.AdamOptimizer(lr=0.01)
        lossed_adam = self.do_linear_optimization(opt2, epochs=7000, do_assert=True)

    # @unittest.skip("This will iterate over 50,000 times .. ")
    def do_linear_optimization(self, optim_func, epochs=25000, batch_size=8,
                               do_assert=True):
        np.random.seed(100)
        x_node = node.VarNode('x')
        y_node = node.VarNode('y')

        net_w = np.array([[-1, -3, 1], [0, 4, -2]])
        net_b = np.array([3, -2]).reshape((2, 1))

        dense = node.DenseLayer(x_node, 2, net_w, net_b)
        l2_node = L2DistanceSquaredNorm(dense, y_node)

        # optim_func = self.rate_adjustable_optimizer_func(0.01)
        # adam = core.np.Optimization.AdamOptimizer()
        optimizer = core.np.Optimization.OptimizerIterator([x_node, y_node], l2_node, optim_func)
        log_at_info()
        epoch = 0
        losses = []
        for x, y in self.model.data(epochs, batch_size):
            var_map = {'x': x, 'y': y}
            loss = optimizer.step(var_map, 1.0)
            #losses.append(loss)
            if epoch % 100 == 0:
                losses.append([epoch, loss])
            if epoch % 1000 == 0:
                info("[{}] Loss:{}".format(epoch, loss))
            epoch += 1
        info("[{}] Loss:{}".format(epoch, loss))

        dense_w = dense.get_w()
        dense_b = dense.get_b()
        info("w = np.{}".format(repr(dense_w)))
        info("b = np.{}".format(repr(dense_b)))
        if do_assert:
            np.testing.assert_array_almost_equal(dense_w, self.model_w, 3)
            np.testing.assert_array_almost_equal(dense_b, self.model_b, 3)
        return np.array(losses)

    def plot_losses(self, losses, x_axis=None, av=None):
        if x_axis:
            plt.scatter(x_axis, losses)
        else:
            plt.plot(losses)
        if not av:
            plt.show()
            return
        av_np = np.array(av)
        plt.plot(av_np[:, 0], av_np[:, 1], 'r')
        plt.show()


if __name__ == '__main__':
    unittest.main()
