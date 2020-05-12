import numpy as np

from core.np.Nodes import BinaryMatrixOp


class L2DistanceSquaredNorm(BinaryMatrixOp):
    r"""
    y_pre and y_actual should both be N x 1 matrices but there are no
    checks at present
    """

    def __init__(self, y_predicted, y_actual, name=None):
        BinaryMatrixOp.__init__(self, y_predicted, y_actual, name)

    def _do_compute(self, var_map):
        y_pred = self.a_node.value(var_map)
        y_act = self.b_node.value(var_map)
        y_pred = y_pred.reshape((-1,))
        y_act = y_act.reshape((-1,))
        y_del = y_pred - y_act
        return np.sum(np.square(y_del)) / y_del.size

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        y_pred = self.a_node.value(var_map)
        y_act = self.b_node.value(var_map)
        y_del = 2 * (y_pred - y_act)
        y_pred_grad = y_del * self._grad_value
        y_act_grad = -y_del * self._grad_value
        self.a_node.backward(y_pred_grad, self, var_map, tab + " ")
        self.b_node.backward(y_act_grad, self, var_map, tab + " ")


class SoftmaxLoss(BinaryMatrixOp):
    def __init__(self, pred, target, name=None):
        r"""

        :param pred:  mxn shape where m is output dimension (categories) and n is the number of instances
        in the batch
        :param target:  mxn one hot vectors where m in number of categories along 0 axis, and n is the number
        of instances in the batch (axis=1)
        :param name:
        """
        BinaryMatrixOp.__init__(pred, target, name)

    def _prediction_node(self):
        return self.a_node

    def _target_node(self):
        return self.b_node

    def _do_compute(self, var_map):
        y_p = self._prediction_node().value(var_map)
        y_t = self._target_node().value(var_map)
        self.y_p_exp = np.exp(y_p)
        self.y_p_norm = np.sum(self.y_p_exp, axis=0)
        self.softmax = self.y_p_exp / self.y_p_norm
        self.correct_rows = (y_t > 0).argmax(axis=0)
        return self.softmax * y_t

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        pass
