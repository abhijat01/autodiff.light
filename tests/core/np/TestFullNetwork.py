import unittest
import numpy as np
import core.np.Nodes as node
import math
import matplotlib.pyplot as plt
import tests.core.np.TestModels as models


class FullNetworkWithSigmoid(unittest.TestCase):
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
        wx_node = node.LinearTransform(w_node, x_node)
        sum_node = node.MatrixAddition(wx_node, b_node)
        sigmoid_node = node.SigmoidNode(sum_node)
        l2_node = node.L2DistanceSquaredNorm(sigmoid_node, ya_node)

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

    def get_model_generator(self):
        #return models.Parabola()
        return models.Transform1()

    def test_transform1(self):
        model = models.Transform1()

        def optimizer_function(_w, grad):
            return _w - 0.001 * grad
        self.run_model(model, optimizer_function, 15000)

    def test_parabola(self):
        model = models.Parabola()

        def optimizer_function(_w, grad):
            return _w - 0.001 * grad
        self.run_model(model, optimizer_function, 15000)

    def run_model(self, model, optimizer_func, steps):
        var_map, start_nodes, l2_node = models.make__two_layer_model()
        optimizer = node.OptimizerIterator(start_nodes, l2_node, optimizer_func)
        node.OptimizerIterator.set_log_to_info()
        count = 0
        losses = []
        sum_losses = 0
        av = []
        x_axis = []
        for (x, y) in model.data(steps):
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

        print(var_map['w'])
        print("-------------")
        print(var_map['b'])
        print("---- w2 ----- ")
        print(var_map['w2'])
        print("----- b2 ----")
        print(var_map['b2'])

        print("[{}] Current loss:{} Average loss so far:{}".format(count, loss, average_l100))
        plt.scatter(x_axis, losses)
        av_np = np.array(av)
        plt.plot(av_np[:,0], av_np[:,1], 'r')
        plt.show()

    def simple_name(self):
        return FullNetworkWithSigmoid.__name__


if __name__ == '__main__':
    unittest.main()
