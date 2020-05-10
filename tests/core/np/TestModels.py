import numpy as np
import core.np.Nodes as node
import math
import random
import matplotlib.pyplot as plt


def make__two_layer_model():
    r"""
    Designed to be used only with TestFullNetwork class.. variable names etc. are
    used in debugging in the testing code
    """
    w_node = node.VarNode('w', True)
    x_node = node.VarNode('x')
    ya_node = node.VarNode('y_a')
    b_node = node.VarNode('b', True)
    w2_node = node.VarNode('w2', True)
    b2_node = node.VarNode('b2', True)
    start_nodes = [w_node, x_node, b_node, ya_node, w2_node, b2_node]

    w = np.array([[1, 3], [0, 1]])
    x = (np.array([[1, -1, 2]])).T
    b = np.array([[-2, -3]]).T
    y_act = np.array([[.5, .7]]).T
    w2 = np.array([[.1, .2], [.3, .07]])
    b2 = np.array([[.02, .3]]).T
    var_map = {'w': w, 'x': x, 'y_a': y_act, 'b': b, 'w2': w2, 'b2': b2}

    wx_node = node.MatrixMultiplication(w_node, x_node, "wx")
    sum_node = node.MatrixAddition(wx_node, b_node, "wx+b")
    sigmoid_node = node.SigmoidNode(sum_node, "sig")

    wx2_node = node.MatrixMultiplication(w2_node, sigmoid_node, "wx2")
    sum2_node = node.MatrixAddition(wx2_node, b2_node, "wx2+b2")

    l2_node = node.L2DistanceSquaredNorm(sum2_node, ya_node, "l2")
    return var_map, start_nodes, l2_node


class LinearModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.input_dim = self.w.shape[1]

    def data(self, epochs, batch_size=1):
        np.random.seed(100)
        epoch_count = 0
        while epoch_count < epochs:
            epoch_count += 1
            x = np.random.rand(self.input_dim, batch_size)
            y = self.w @ x + self.b
            yield x, y


class Transform1:
    def __init__(self):
        pass

    def x(self, x1, x2):
        return x1 * x1 - x2 * x2

    def y(self, x1, x2):
        return 2 * x1 * x2

    def data(self, count, batch_size=1):
        total = 0
        while total < count:
            total += 1
            x1 = random.uniform(-1, 1)
            x2 = random.uniform(-1, 1)
            train = np.array([x1, x2]).reshape((2, 1))
            target = np.array([self.x(x1, x2), self.y(x1, x2)]).reshape((2, 1))
            yield train, target


class Parabola:
    def __init__(self):
        self.theta = np.arange(-1, 1, 0.001)
        x = np.array([
            [t for t in self.theta],
            [(4 * t) ** 2 for t in self.theta]
        ])
        np.random.seed(100)
        random_noise = np.random.uniform(-0.02, 0.02, x.shape)
        self.learning_data = x + random_noise
        random_noise = np.random.uniform(-0.02, 0.02, x.shape)
        self.ground_truth = x + random_noise

    def do_plot(self):
        plt.ylim(0, 1)
        plt.xlim(-.6, .6)
        plt.scatter(self.learning_data[0], self.learning_data[1])
        plt.plot(self.ground_truth[0], self.ground_truth[1], 'r')
        plt.ylabel('some numbers')
        plt.show()

    def data(self, count, batch_size=1):
        np.random.seed(100)
        total = 0
        while total < count:
            idx = np.random.randint(len(self.theta), size=batch_size)
            # idx = np.random.choice(len(self.theta))
            x = self.learning_data[:, idx]  # .reshape((2, 1))
            y = self.ground_truth[:, idx]  # .reshape((2, 1))
            total += 1
            yield x, y
