import numpy as np

from core.np.Nodes import BinaryMatrixOp, MComputeNode


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
