import unittest
import tests.core.np.BaseTests as base
import numpy as np
import core.np.Nodes as node
import core.np.regularization as reg
from core import debug, info


class BatchNormalizationTest(base.BaseComputeNodeTest):

    def test_batch_norm(self):
        x = np.array([[1, 2, 3], [3, 4, 1]])
        x_node = node.VarNode('x')
        bnorm = reg.BatchNormalization(x_node)
        x_node.forward({'x': x})
        norm_value = bnorm.value()
        debug("normalized value = np.{}".format(repr(norm_value)))
        bnorm.backward(np.ones_like(norm_value), self, {'x': x})
        grad_at_x = x_node.total_incoming_gradient()
        debug("grad_at_x = np.{}".format(repr(grad_at_x)))
        debug("[BatchNormalizationTest.test_something()] bnorm.gamma_grad = np.{}".format(repr(bnorm.gamma_grad)))
        debug("[BatchNormalizationTest.test_something()] bnorm.beta_grad = np.{}".format(repr(bnorm.beta_grad)))

    def test_dropout_simple_input(self):
        x = np.array([[1, 2, 3], [3, 4, 1]])
        x_node = node.VarNode('x')
        dropout = reg.Dropout(x_node)
        ctx = node.ComputeContext({'x': x})
        x_node.forward(ctx)
        value = dropout.value()
        info("[BatchNormalizationTest.test_dropout_simple_input()] input   value  = np.{}".format(repr(x)))
        info("[BatchNormalizationTest.test_dropout_simple_input()] dropout value = np.{}".format(repr(value)))

    def test_dropout_with_dense(self):
        model_w = np.array([[1, 3, -1], [0, -4, 2.]])
        model_b = np.array([-3, 2.]).reshape((2, 1))
        x_node = node.VarNode('x')
        dense = node.DenseLayer(x_node, output_dim=2, initial_w=model_w, initial_b=model_b)
        p = .6
        dropout = reg.Dropout(dense, dropout_prob=p)
        x = np.array([[1, -1], [2, 3], [-1, -2.]])
        ctx = node.ComputeContext({'x': x})
        found_0 = False
        count = 0
        row_to_check = 0
        while not found_0:
            x_node.forward(ctx)
            output = dropout.value()
            sq = np.sum(np.square(output), axis=1)
            found_0 = sq[row_to_check] == 0
            count += 1
            if count > 100:
                raise Exception("Could not get 0's in first row after {} iterations.".format(count))

        info("[DenseLayerStandAlone.test_single_step()] output = np.{}".format(repr(output)))
        dropout.backward(np.ones_like(output), self, ctx)
        w_grad = dense.get_w_grad()
        info("[DenseLayerStandAlone.test_single_step()] w_grad = np.{}".format(repr(w_grad)))
        b_grad = dense.get_b_grad()
        info("[DenseLayerStandAlone.test_single_step()] b_grad = np.{}".format(repr(b_grad)))
        wg_sq_sum = np.sum(np.square(w_grad), axis=1)
        self.assertEqual(0,wg_sq_sum[row_to_check])
        bg_sum_sq = np.sum(np.square(b_grad), axis=1)
        self.assertEqual(0, bg_sum_sq[row_to_check])

        # Test validation time (not training time)
        ctx.set_is_training(False)
        x_node.forward(ctx)
        test_output = dropout.value()
        np.testing.assert_array_almost_equal(test_output, dense.value()*p)


if __name__ == '__main__':
    unittest.main()
