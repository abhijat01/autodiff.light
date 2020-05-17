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
        y_pred = self.a_node.value()
        y_act = self.b_node.value()
        y_del = y_pred - y_act
        return np.sum(np.square(y_del)) / y_del.size

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        y_pred = self.a_node.value()
        y_act = self.b_node.value()
        y_del = 2 * (y_pred - y_act)
        y_pred_grad = (y_del * self._grad_value) # /y_pred.size
        y_act_grad = -(y_del * self._grad_value) #/y_pred.size
        self.a_node.backward(y_pred_grad, self, var_map)
        self.b_node.backward(y_act_grad, self, var_map)


class CrossEntropy(MComputeNode):
    def __init__(self, predicted, target, name=None):
        MComputeNode.__init__(self, name)
        self.predicted = predicted
        self.target = target
        self._add_upstream_nodes([predicted, target])
        self.predicted_logs = None
        self.predicted_fixed = None

    def forward(self, var_map):
        self.fwd_count += 1
        if not self.can_go_fwd():
            return
        yp = self.predicted.value()
        yt = self.target.value()
        yp[yp == 0] = 1e-10
        self.predicted_fixed = yp
        self.predicted_logs = -np.log(yp)
        self.node_value = np.sum(self.predicted_logs * yt)
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        grad_to_predicted = -np.reciprocal(self.predicted_fixed) * self.target.value()
        grad_to_target = self.predicted_logs
        self.predicted.backward(grad_to_predicted, self, var_map)
        self.target.backward(grad_to_target, self, var_map)
