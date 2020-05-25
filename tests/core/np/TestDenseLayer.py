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
        y_target = node.VarNode('y_target')

        dense = node.DenseLayer(x_node, 2, self.model_w, self.model_b)
        l2_node = L2DistanceSquaredNorm(dense, y_target)

        var_map = {'x': x, 'y_target': y}
        x_node.forward(var_map)
        y_target.forward(var_map)

        log_at_info()
        value = dense.value()
        info("------------------------------------------")
        info("Predicted value = np.{}".format(repr(value)))
        info("Target    value = np.{}".format(repr(y)))
        value = l2_node.value()
        info("L2 node value (loss):{}".format(value))

        info("------------------------------------------")
        info("Printing weights (not updated yet)")
        info("------------------------------------------")
        info("Linear layer weight = np.{}".format(repr(dense.get_w())))
        info("Linear layer bias   = np.{}".format(repr(dense.get_b())))

        optim_func = self.rate_adjustable_optimizer_func(0.001)

        optimizer = core.np.Optimization.OptimizerIterator([x_node, y_target], l2_node, optim_func)
        optimizer.step(var_map, 1.0)
        np.set_printoptions(precision=64, floatmode='maxprec_equal')
        info("------------------------------------------")
        info("Printing after updating weights")
        info("------------------------------------------")
        info("Linear layer weight:{}".format(repr(dense.get_w())))
        info("Linear layer bias:{}".format(repr(dense.get_b())))
        info("w_grad = np.{}".format(repr(dense.get_w_grad())))
        info("b_grad = np.{}".format(repr(dense.get_b_grad())))
        expected_weight = np.array([[1.0000, 2.9850, -0.9910],
                                    [-0.0040, -3.9755, 1.9845]])
        expected_bias = np.array([[-3.006],
                                  [2.009]])
        expected_w_grad = np.array([[0.0, 15.0, -9.0],
                                    [4.0, -24.5, 15.5]])
        expected_b_grad = np.array([[6.],
                                    [-9.]])

        np.testing.assert_almost_equal(expected_weight, dense.get_w())
        np.testing.assert_almost_equal(expected_w_grad, dense.get_w_grad())
        np.testing.assert_almost_equal(expected_bias, dense.get_b())
        np.testing.assert_almost_equal(expected_b_grad, dense.get_b_grad())

    def test_basic_op_large_matrix(self):
        r"""
        Runs test for a slightly larger matrix
        :return:
        """
        x = np.array([[0.54566752, 0.66921034, 0.35265542, 0.32324271, 0.35036963,
                       0.05317591],
                      [0.97433629, 0.5027976, 0.15637831, 0.72948084, 0.42097552,
                       0.52522781],
                      [0.41793729, 0.48112345, 0.46862087, 0.88918467, 0.48792933,
                       0.32439625],
                      [0.4775774, 0.58105899, 0.35079832, 0.79657794, 0.3910011,
                       0.72908915]])
        w = np.array([[0.61013274, 0.86914947, 0.95211922, 0.96385655],
                      [0.64290252, 0.2717017, 0.193146, 0.05004571],
                      [0.14360354, 0.54256991, 0.90870491, 0.06577582]])
        b = np.array([[0.76026806],
                      [0.32982798],
                      [0.01258297]])
        pred = w @ x + b
        target = np.ones_like(pred)

        x_node = node.VarNode('x')
        target_node = node.VarNode('y_target')

        dense = node.DenseLayer(x_node, 3, w, b)
        l2_node = L2DistanceSquaredNorm(dense, target_node)

        var_map = {'x': x, 'y_target': target}
        x_node.forward(var_map)
        target_node.forward(var_map)

        log_at_info()
        predicted = dense.value()
        info("------------------------------------------")
        info("Predicted value = np.{}".format(repr(predicted)))
        info("Target    value = np.{}".format(repr(target)))
        loss = l2_node.value()
        info("L2 node value (loss):{}".format(loss))

        info("------------------------------------------")
        info("Printing weights (not updated yet)")
        info("------------------------------------------")
        info("Linear layer weight = np.{}".format(repr(dense.get_w())))
        info("Linear layer bias   = np.{}".format(repr(dense.get_b())))

        optim_func = self.rate_adjustable_optimizer_func(0.001)

        optimizer = core.np.Optimization.OptimizerIterator([x_node, target_node], l2_node, optim_func)
        optimizer.step(var_map, 1.0)
        np.set_printoptions(precision=64, floatmode='maxprec_equal')
        info("------------------------------------------")
        info("Printing after updating weights")
        info("------------------------------------------")
        info("weight=np.{}".format(repr(dense.get_w())))
        info("w_grad = np.{}".format(repr(dense.get_w_grad())))
        info("bias = np.{}".format(repr(dense.get_b())))
        info("b_grad = np.{}".format(repr(dense.get_b_grad())))

        # These are values from pytorch
        expected_weight = np.array([[0.60973525, 0.86854088, 0.95157486, 0.96327269],
                                    [0.64292222, 0.27173772, 0.19318908, 0.05009926],
                                    [0.14362818, 0.54258782, 0.90872669, 0.06581017]])
        expected_w_grad = np.array([[0.39752683, 0.60859025, 0.54437733, 0.58387089],
                                    [-0.01970989, -0.03603142, -0.04307830, -0.05355303],
                                    [-0.02465229, -0.01786957, -0.02174304, -0.03434603]])
        expected_bias = np.array([[0.75927186, 0.32992661, 0.01267095]]).T
        expected_b_grad = np.array([[0.99619532, -0.09862594, -0.08797690]]).T

        np.testing.assert_almost_equal(expected_weight, dense.get_w())
        np.testing.assert_almost_equal(expected_w_grad, dense.get_w_grad())
        np.testing.assert_almost_equal(expected_bias, dense.get_b())
        np.testing.assert_almost_equal(expected_b_grad, dense.get_b_grad())

    def step_and_compare(self, expected_weight, expected_weight_grad, expected_bias, expected_bias_grad):
        pass

    @unittest.skip("This should be moved to notebook")
    def test_compare_optimizations(self):
        opt1 = self.rate_adjustable_optimizer_func(0.01)
        losses_gd = self.do_linear_optimization(opt1, epochs=20000, do_assert=True)

        opt2 = core.np.Optimization.AdamOptimizer(lr=0.01)
        losses_adam = self.do_linear_optimization(opt2, epochs=20000, do_assert=True)

        start_idx = 100
        plt.plot(losses_gd[start_idx:, 0], losses_gd[start_idx:, 1], 'r')
        plt.plot(losses_adam[start_idx:, 0], losses_adam[start_idx:, 1], 'g')
        plt.show()

    def test_linear_optimization(self):
        adam = core.np.Optimization.AdamOptimizer(lr=0.01)
        self.do_linear_optimization(adam, epochs=7000, do_assert=True)

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
            # losses.append(loss)
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

    def test_single_step(self):
        model_w = np.array([[1, 3, -1], [0, -4, 2.]])
        model_b = np.array([-3, 2.]).reshape((2, 1))
        x_node = node.VarNode('x')
        dense = node.DenseLayer(x_node, output_dim=2, initial_w=model_w, initial_b=model_b)
        x = np.array([[1, -1], [2, 3], [-1, -2.]])
        ctx = node.ComputeContext({'x':x})
        x_node.forward(ctx) 
        output = dense.value() 
        info("[DenseLayerStandAlone.test_single_step()] output = np.{}".format(repr(output)))
        dense.backward(np.ones_like(output), self, ctx)
        w_grad = dense.get_w_grad()
        info("[DenseLayerStandAlone.test_single_step()] w_grad = np.{}".format(repr(w_grad)))
        b_grad = dense.get_b_grad()
        info("[DenseLayerStandAlone.test_single_step()] b_grad = np.{}".format(repr(b_grad)))


if __name__ == '__main__':
    unittest.main()
