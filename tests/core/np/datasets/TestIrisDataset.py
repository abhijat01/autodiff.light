import core.np.Optimization
from core.np.datasets.IrisDataset import Iris
from core import debug, info, log_at_info
import core.np.Nodes as node
import core.np.Loss as loss
import core.np.Activations as act
from tests.core.np.BaseTests import BaseComputeNodeTest
import numpy as np
import core.np.utils as utils


class IrisDsTest(BaseComputeNodeTest):

    def test_load(self):
        iris = Iris()
        iris.split_test_train()
        for x, y in iris.train_iterator(3, 1, one_hot=False):
            debug("x = np.{}, y={}".format(repr(x.T), y))

    def test_linear_fit(self):
        epochs = 2000
        iris = Iris()
        x_node = node.VarNode('x')
        yt_node = node.VarNode('yt')
        dense = node.DenseLayer(x_node, 3)
        softmax = act.Softmax(dense)
        cross_entropy = loss.CrossEntropy(softmax, yt_node)
        optimizer_func = core.np.Optimization.AdamOptimizer(lr=0.001)
        optimizer = core.np.Optimization.OptimizerIterator([x_node, yt_node], cross_entropy, optimizer_func)
        log_at_info()
        epoch = 0
        ctx = node.ComputeContext()
        for x, y in iris.train_iterator(epochs, 8):
            ctx['x'] , ctx['yt'] = x, y
            loss_now = optimizer.step(ctx, 1.0)
            if epoch % 100 == 0:
                info("[{}]\tloss_now = {}".format(epoch, loss_now))
            epoch += 1

        f = node.make_evaluator([x_node, yt_node], softmax)
        total, correct = 40, 0
        for x, y_actual in iris.test_iterator(total, one_hot=False):
            ctx['x'], ctx['yt'] = x, y_actual
            y_predicted = f.at(ctx)
            max_idx = np.argmax(y_predicted)
            if max_idx == y_actual:
                correct += 1
        percent = correct * 100 / total
        print("Correct= {}%".format(percent))

    def test_multi_layer(self):
        r"""
        This actually performs better with SGD and normal initialization.
        Gets almost 99% with SGD and normal initialization
        :return:
        """

        iris = Iris()
        x_node = node.VarNode('x')
        yt_node = node.VarNode('yt')
        dense = node.DenseLayer(x_node, 16)
        tanh = act.TanhNode(dense)

        dense2 = node.DenseLayer(tanh, 10)
        relu = act.RelUNode(dense2)

        dense3 = node.DenseLayer(relu, 3)
        softmax = act.Softmax(dense3)

        cross_entropy = loss.CrossEntropy(softmax, yt_node)
        #optimizer_func = core.np.Optimization.AdamOptimizer()
        optimizer_func = core.np.Optimization.SGDOptimizer(lr=0.01)
        optimizer = core.np.Optimization.OptimizerIterator([x_node, yt_node], cross_entropy, optimizer_func)
        log_at_info()

        epoch = 0
        epochs = 10000
        batch_size = 8
        ctx = node.ComputeContext(weight_initializer=None)
        for x, y in iris.train_iterator(epochs, batch_size):
            ctx['x'] , ctx['yt']= x, y
            loss_now = optimizer.step(ctx, 1.0) / batch_size
            if epoch % 500 == 0:
                info("[{}]\tloss_now = {}".format(epoch, loss_now))
            epoch += 1

        f = node.make_evaluator([x_node, yt_node], softmax)
        total, correct = 100, 0
        for x, y_actual in iris.test_iterator(total, one_hot=False):
            var_map = {'x': x, 'yt': y_actual}
            y_predicted = f(var_map)
            max_idx = np.argmax(y_predicted)
            mark = 'x'
            if max_idx == y_actual:
                correct += 1
                mark = u'\u2713'
            print("X:{}, y_pred:{}, Actual={}, Predicted:{}  {}".format(x.T, y_predicted.T, y_actual[0], max_idx, mark))
        percent = correct * 100 / total
        print("Correct= {}%".format(percent))
        self.assertTrue(percent > 95)
