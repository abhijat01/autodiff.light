import unittest
import numpy as np

from core.np.Loss import L2DistanceSquaredNorm
import core.np.Nodes as node
import matplotlib.pyplot as plt

from  core.np.Activations import SigmoidNode
import tests.core.np.TestModels as models
from core import info
from . import BaseComputeNodeTest


class FullNetworkWithSigmoid(BaseComputeNodeTest):
    def test_full_sgmoid_node(self):
        w_node = node.VarNode('w', True)
        x_node = node.VarNode('x')
        ya_node = node.VarNode('y_a')
        b_node = node.VarNode('b', True)
        start_nodes = [w_node, x_node, b_node, ya_node]

        w = np.array([[1, 3, 0], [0, 1, -1]])
        x = (np.array([[1, -1, 2]])).T
        b = np.array([[-2, -3]]).T
        y_act = np.array([[.5, .7]]).T
        var_map = {'w': w, 'x': x, 'y_a': y_act, 'b': b}
        wx_node = node.MatrixMultiplication(w_node, x_node)
        sum_node = node.MatrixAddition(wx_node, b_node)
        sigmoid_node = SigmoidNode(sum_node)
        l2_node = L2DistanceSquaredNorm(sigmoid_node, ya_node)

        def default_optimizer_function(_w, grad, lr=1):
            return _w - lr * grad

        optimizer = node.OptimizerIterator(start_nodes, l2_node, default_optimizer_function)
        node.OptimizerIterator.set_log_to_info()
        losses = []
        for i in range(100):
            loss = optimizer.step(var_map, 1.0)
            losses.append(loss)
            if i % 10 == 0:
                print("[{}] Loss:{}".format(i, loss))
        print("Final loss:{}".format(loss))
        print("w:{}".format(var_map['w']))
        print("b:{}".format(var_map['b']))

    def test_transform1(self):
        model = models.Transform1()
        # change this to 10,000 when running real test
        iter_count = 1500
        def optimizer_function(_w, grad):
            return _w - 0.001 * grad
        self.run_model(model, optimizer_function, iter_count)

    def test_parabola(self):
        model = models.Parabola()
        # Change this to 20,000 when running real test
        iter_count = 1500

        def optimizer_function(_w, grad):
            return _w - 0.001 * grad
        self.run_model(model, optimizer_function, iter_count)

    def run_model(self, model, optimizer_func, epochs):
        var_map, start_nodes, l2_node = models.make__two_layer_model()
        optimizer = node.OptimizerIterator(start_nodes, l2_node, optimizer_func)
        node.OptimizerIterator.set_log_to_info()
        count = 0
        losses = []
        sum_losses = 0
        av = []
        x_axis = []
        for (x, y) in model.data(epochs, 2):
            # print("count:{}".format(count))
            var_map['x'] = x
            var_map['y_a'] = y
            loss = optimizer.step(var_map, 1.0)
            losses.append(loss)
            x_axis.append(count)
            sum_losses += loss
            if count % 500 == 0:
                last_100 = losses[-100:]
                average_l100 = sum(last_100) / len(last_100)
                av.append([count, average_l100])
                print("[{}] Current loss:{} Average loss so far:{}".format(count, loss, average_l100))

            count += 1

        last_100 = losses[-100:]
        average_l100 = sum(last_100) / len(last_100)
        av.append([count, average_l100])

        info("Now printing w and b ..W:")
        info(var_map['w'])
        info("-------------b:")
        info(var_map['b'])
        info("---- print w2 and b2...  W2:")
        info(var_map['w2'])
        info("----- b2 ----")
        info(var_map['b2'])

        info("[{}] Current loss:{} Average loss so far:{}".format(count, loss, average_l100))
        plt.scatter(x_axis, losses)
        av_np = np.array(av)
        plt.plot(av_np[:,0], av_np[:,1], 'r')
        plt.show()


if __name__ == '__main__':
    unittest.main()
