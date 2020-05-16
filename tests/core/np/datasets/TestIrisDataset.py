import unittest

import core.np.Optimization
from core.np.datasets.IrisDataset import Iris
from core import debug, info, log_at_info
import core.np.Nodes as node
import core.np.Loss as loss
import core.np.Activations as act
from tests.core.np.BaseTests import BaseComputeNodeTest
import numpy as np


class IrisDsTest(BaseComputeNodeTest):

    def test_load(self):
        iris = Iris()
        iris.split_test_train()
        for x, y in iris.train_iterator(3, 3):
            debug("x = np.{}".format(repr(x)))
            debug("y = np.{}".format(repr(y)))

    def test_linear_fit(self):
        epochs = 2000
        iris = Iris()
        x_node = node.VarNode('x')
        yt_node = node.VarNode('yt')
        dense = node.DenseLayer(x_node, 3)
        softmax = act.Softmax( dense)
        cross_entropy = loss.CrossEntropy(softmax, yt_node)
        optimizer_func = core.np.Optimization.AdamOptimizer(lr=0.001)
        optimizer = core.np.Optimization.OptimizerIterator([x_node, yt_node], cross_entropy, optimizer_func)
        log_at_info()
        epoch = 0
        for x, y in iris.train_iterator(epochs, 8):
            var_map = {'x': x, 'yt': y}
            loss_now = optimizer.step(var_map, 1.0)
            if epoch % 100 == 0:
                info("[{}]\tloss_now = {}".format(epoch, loss_now))
            epoch += 1

        f = node.make_evaluator([x_node, yt_node], softmax)
        total, correct = 40, 0
        for x, y_actual in iris.test_iterator(total, one_hot=False):
            var_map = {'x': x, 'yt': y_actual}
            y_predicted = f.at(var_map)
            max_idx = np.argmax(y_predicted)
            if max_idx == y_actual:
                correct += 1
        percent = correct * 100 / total
        print("Correct= {}%".format(percent))

    def test_mult_layer(self):

        iris = Iris(.8)
        x_node = node.VarNode('x')
        yt_node = node.VarNode('yt')
        dense = node.DenseLayer(x_node, 16)
        tanh = act.TanhNode(dense)
        dense2 = node.DenseLayer(tanh,10)
        relu = act.RelUNode(dense2)
        dense3 = node.DenseLayer(relu, 3)
        softmax = act.Softmax( dense3)
        cross_entropy = loss.CrossEntropy(softmax, yt_node)
        optimizer_func = core.np.Optimization.AdamOptimizer(lr=0.1)
        optimizer = core.np.Optimization.OptimizerIterator([x_node, yt_node], cross_entropy, optimizer_func)
        log_at_info()
        epoch = 0
        epochs = 5000
        batch_size=8
        for x, y in iris.train_iterator(epochs, batch_size):
            var_map = {'x': x, 'yt': y}
            loss_now = optimizer.step(var_map, 1.0)/batch_size
            if epoch % 100 == 0:
                info("[{}]\tloss_now = {}".format(epoch, loss_now))
            epoch += 1

        f = node.make_evaluator([x_node, yt_node], softmax)
        total, correct= 40, 0
        for x, y_actual in iris.test_iterator(total, one_hot=False):
            var_map = {'x': x, 'yt': y_actual}
            y_predicted = f.at(var_map)
            max_idx = np.argmax(y_predicted)
            if max_idx == y_actual:
                correct+=1
        percent = correct*100/total
        print("Correct= {}%".format(percent))







