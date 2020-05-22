from core.np.datasets.Mnist784Dataset import Mnist784
from core import debug, info, log_at_info, log_at_debug
from tests.core.np.BaseTests import BaseComputeNodeTest
import numpy as np
import matplotlib.pyplot as plt
import core.np.Nodes as node
import core.np.Activations as act
import core.np.Loss as loss
import core.np.Optimization as autodiff_optim
from core.np.utils import to_one_hot

import time


class Mnist784DsTest(BaseComputeNodeTest):
    def test_load_one(self):
        mnist = Mnist784()
        for x, y in mnist.train_iterator(1):
            print(repr(y))
            x_img = x.reshape(28, 28)
            plt.imshow(x_img, cmap='gray', vmin=0, vmax=255)
            plt.show()

    def test_linear_training(self):
        x_node = node.VarNode('x')
        yt_node = node.VarNode('yt')
        linear1 = node.DenseLayer(x_node, 100, name="Dense-First", weight_scale=0.01)
        relu1 = act.RelUNode(linear1, name="RelU-First")
        linear2 = node.DenseLayer(relu1, 200, name="Dense-Second", weight_scale=0.01)
        relu2 = act.RelUNode(linear2, name="RelU-Second")
        linear3 = node.DenseLayer(relu2, 10, name="Dense-Third", weight_scale=0.01)

        cross_entropy = loss.LogitsCrossEntropy(linear3, yt_node, name="XEnt")
        optimizer_func = autodiff_optim.SGDOptimizer(lr=.1)
        #optimizer_func = autodiff_optim.AdamOptimizer(lr=.01)
        optimizer = autodiff_optim.OptimizerIterator([x_node, yt_node], cross_entropy, optimizer_func)
        epochs = 25
        batch_size = 64
        log_at_info()
        last_x = None
        losses = []
        _s = 0
        import tests.core.np.datasets.mnist as mn
        x_train, y_train, x_val, y_val, x_test, y_test = mn.load_dataset(flatten=True)
        iter_count = 1
        f = node.make_evaluator([x_node, yt_node], linear3)
        total_time = time.time()
        for epoch in range(epochs):
            epoch_time = time.time()
            for x, y in iterate_over_minibatches(x_train, y_train, batch_size=batch_size, shuffle=False):
                input_x = x.T
                y_target = to_one_hot(y, max_cat_num=9)
                var_map = {'x': input_x, 'yt': y_target}
                start = time.time()
                iter_loss = optimizer.step(var_map, 1.0) / batch_size
                end = time.time()
                losses.append(iter_loss)
                iter_count += 1
            epoch_time = time.time()-epoch_time
            loss_av = np.array(losses[:-batch_size + 1])
            loss_av = np.mean(loss_av)
            var_map = {'x': x_val.T, 'yt': to_one_hot(y_val)}
            y_predicted = f(var_map)
            arg_max = np.argmax(y_predicted, axis=0)
            correct = arg_max == y_val
            percent = np.mean(correct) * 100
            info("Epoch {:2d}:: Validation accuracy:[{:5.2f}%] loss av={:01.8f}, time:{:2.3f}s".format(epoch, percent, loss_av, epoch_time))

        total_time = time.time() - total_time
        info("[Mnist784DsTest.test_linear_training()] total_time = {:5.3f} s".format(total_time))

    def test_ds(self):
        import tests.core.np.datasets.mnist as mn
        xt, yt, xval, yval, xtest, ytest = mn.load_dataset(flatten=True)
        debug("xt.shape:{}".format(xt.shape))
        debug("yt.shape:{}".format(yt.shape))
        debug("xval.shape:{}".format(xval.shape))
        debug("yval.shape:{}".format(yval.shape))
        debug("[Mnist784DsTest.test_ds()] yt[3] = {}".format(repr(yt[3])))
        for x, y in iterate_over_minibatches(xt, yt, batch_size=32, shuffle=False):
            debug("[Mnist784DsTest.test_ds()] x.shape = np.{}".format(repr(x.shape)))
            debug("[Mnist784DsTest.test_ds()] y.shape = np.{}".format(repr(y.shape)))
        debug("[Mnist784DsTest.test_ds()] np.max(yt) = np.{}".format(repr(np.max(yt))))


def iterate_over_minibatches(inputs, targets, batch_size):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
