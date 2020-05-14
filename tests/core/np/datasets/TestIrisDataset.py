import unittest

import core.np.Optimization
from core.np.datasets.IrisDataset import Iris
from core import debug, info, log_at_info
import core.np.Nodes as node
import core.np.Loss as loss
import core.np.Activations as act
from tests.core.np.BaseTests import BaseComputeNodeTest


class IrisDsTest(BaseComputeNodeTest):

    def test_load(self):
        iris = Iris()
        iris.split_test_train()
        for x, y in iris.train_iterator(3, 3):
            debug("x = np.{}".format(repr(x)))
            debug("y = np.{}".format(repr(y)))

    def test_linear_fit(self):
        epochs = 500
        iris = Iris()
        x_node = node.VarNode('x')
        yt_node = node.VarNode('yt')
        dense = node.DenseLayer(x_node, 3)
        softmax = act.Softmax( dense)
        cross_entropy = loss.CrossEntropy(softmax, yt_node)
        optimizer_func = self.rate_adjustable_optimizer_func()
        optimizer = core.np.Optimization.OptimizerIterator([x_node, yt_node], cross_entropy, optimizer_func)
        log_at_info()
        epoch = 0
        for x, y in iris.train_iterator(epochs, 16):
            var_map = {'x': x, 'yt': y}
            loss_now = optimizer.step(var_map, 1.0)
            if epoch % 100 == 0:
                info("[{}]\tloss_now = {}".format(epoch, loss_now))
                self.learning_rate = self.learning_rate*0.5
            epoch += 1


