import unittest

import core.np.Nodes as n
import core.np.rnn.Recurrent as rnn
import numpy as np
from core import log_at_info, debug, info
from tests.core.np.BaseTests import BaseComputeNodeTest
import os
from tests.core.np.rnn.NamesDs import NameDS
import core.np.Loss as loss


class RnnLayerIntegrationTests(BaseComputeNodeTest):
    def setUp(self):
        x1 = np.array([1, 2, 1]).reshape((3, 1))
        self.var_map = {'x': x1}
        self.h = np.array([0.6, 0.2]).reshape((2, 1))
        w = np.array([[0.63315733, 0.51699569, 0.78251473, 0.94678789, 0.30939115],
                      [0.12741137, 0.67238871, 0.23514442, 0.50932127, 0.60643467],
                      [0.26004482, 0.02306102, 0.56403955, 0.32862147, 0.13988205],
                      [0.97815493, 0.66425931, 0.85988497, 0.13528022, 0.03943312]])
        wb = np.array([[0.5],
                       [-0.25],
                       [1.],
                       [-1.]])
        u = np.array([[0.39865366, 0.49334758, 0.29215267, 0.97590111, 0.68403036],
                      [0.03237844, 0.73579572, 0.49288022, 0.32059863, 0.69219668]])
        ub = np.array([[-1],
                       [.5]])
        self.w_param, self.wb_param = rnn.SharedParam(w), rnn.SharedParam(wb)
        self.u_param, self.ub_param = rnn.SharedParam(u), rnn.SharedParam(ub)
        data_dir = os.path.expanduser("~")
        self.data_dir = os.path.join(data_dir, "ai-ws", "rnn-basics", "data", "names")
        self.name_ds = NameDS(self.data_dir)

    def test_rnn_layer(self):
        x = np.array([[1, 2, 1], [-1, 0, -.5]]).T
        x = x.reshape((3, 1, 2))
        input_node = n.VarNode('x')
        var_map = {'x':x}
        rnn_layer = rnn.SimpleRnnLayer(input_node,4,2)
        input_node.forward(var_map)
        y = rnn_layer.value()
        dely = y*.1
        rnn_layer.backward(dely, self, var_map)
        x_grad = input_node.total_incoming_gradient()
        debug("[SimpleRnnCellTests.test_rnn_layer()] x_grad = np.{}".format(repr(x_grad)))

    def test_rnn_layer_with_loss(self):
        debug("[RnnLayerFullTests.test_rnn_layer_with_loss()] self.data_dir = {}".format(self.data_dir))
        x = self.name_ds.line_to_numpy('ABCD')
        debug("[RnnLayerFullTests.test_rnn_layer_with_loss()] ABCD: x = np.{}".format(repr(x)))
        debug("------------------------------------------------------")
        x = self.name_ds.line_to_numpy('Albert')
        debug("[RnnLayerFullTests.test_rnn_layer_with_loss()] x = np.{}".format(repr(x)))
        debug("------------------------------------------------------")
        log_at_info()
        for i in range(5):
            c, l, category_index, name_tensor = self.name_ds.random_training_example()
            debug("[{}]:{}".format(c, l))
            cat_tensor = self.name_ds.category_idx_to_tensor([category_index])
            debug("[RnnLayerFullTests.test_rnn_layer_with_loss()] cat_tensor = np.{}".format(repr(cat_tensor)))

        x_node = n.VarNode('x')
        y_target_node = n.VarNode('y_target')

        ctx = n.ComputeContext({'x': name_tensor, 'y_target': cat_tensor})
        rnn_node = rnn.SimpleRnnLayer(x_node, self.name_ds.n_categories, 128)
        loss_node = loss.LogitsCrossEntropy(rnn_node, y_target_node)

        x_node.forward(ctx)
        y_target_node.forward(ctx)
        y = rnn_node.value()
        info("[RnnLayerFullTests.test_rnn_layer_with_loss()]  y = np.{}".format(repr(y)))
        loss_value = loss_node.value()
        info("[RnnLayerFullTests.test_rnn_layer_with_loss()] loss = np.{}".format(repr(loss_value)))
        loss_node.backward(1.0, self, ctx)
        grads = rnn_node.total_incoming_gradient()
        info(grads)


if __name__ == '__main__':
    unittest.main()
