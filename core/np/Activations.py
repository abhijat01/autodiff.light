import numpy as np
from core.np.Nodes import MComputeNode


class SigmoidNode(MComputeNode):
    def __init__(self, upstream_node, name=None):
        MComputeNode.__init__(self, name, False)
        self.input_node = upstream_node
        self._add_upstream_nodes([upstream_node])

    def forward(self, var_map):
        matrix = self.input_node.value()
        self.node_value = 1 / (1 + np.exp(-matrix))
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        sig_grad = self.node_value * (1 - self.node_value)
        grad_to_input = sig_grad * self.grad_value()
        self.input_node.backward(grad_to_input, self, var_map)


class TanhNode(MComputeNode):
    def __init__(self, upstream_node, name=None):
        MComputeNode.__init__(self, name, False)
        self.input_node = upstream_node
        self._add_upstream_nodes([upstream_node])

    def forward(self, var_map):
        matrix = self.input_node.value()
        self.node_value = np.tanh(matrix)
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        sig_grad = 1 - np.square(self.node_value)
        grad_to_input = sig_grad * self.grad_value()
        self.input_node.backward(grad_to_input, self, var_map)


class RelUNode(MComputeNode):
    def __init__(self, upstream_node, name=None):
        MComputeNode.__init__(self, name, False)
        self.input_node = upstream_node
        self._add_upstream_nodes([upstream_node])

    def forward(self, var_map):

        input_value = self.input_node.value()
        zeros = np.zeros_like(input_value)
        self.node_value = np.maximum(input_value, zeros)
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        input_value = self.input_node.value()
        grad_to_input = self._grad_value * (input_value > 0)
        self.input_node.backward(grad_to_input, self, var_map)


class Softmax(MComputeNode):
    def __init__(self, input_node, name=None):
        r"""

        :param pred:  mxn shape where m is output dimension (categories) and n is the number of instances
        in the batch
        :param target:  mxn one hot vectors where m in number of categories along 0 axis, and n is the number
        of instances in the batch (axis=1)
        :param name:
        """
        MComputeNode.__init__(self, name)
        self.input_node = input_node
        self._add_upstream_nodes([input_node])
        self.epsilon = 1e-15

    def forward(self, var_map):
        y = self.input_node.value()
        y = y - np.max(y)
        e = np.exp(y) + self.epsilon
        s = np.sum(e, axis=0)
        self.node_value = e / s
        self._forward_downstream(var_map)

    def _l2_norm(self, vec):
        sq = np.square(vec)
        sums = np.sum(sq, axis=0)
        return vec / sums

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        grad_value_to_use = self._grad_value
        if np.max(grad_value_to_use) > 10:
            grad_value_to_use = self._l2_norm(grad_value_to_use)

        categories = self.node_value.shape[0]
        num_batches = self.node_value.shape[1]
        grad_to_input = np.zeros((categories, num_batches))
        for d in range(num_batches):
            batch_grad = np.zeros((categories, categories))
            for i in range(categories):
                for j in range(categories):
                    if i == j:
                        batch_grad[i, j] = self.node_value[i, d] * (1 - self.node_value[i, d])
                    else:
                        batch_grad[i, j] = -self.node_value[i, d] * self.node_value[j, d]
            grad_to_input[:, d] = batch_grad @ grad_value_to_use[:, d]
        grad_to_input = grad_to_input / grad_to_input.size
        self.input_node.backward(grad_to_input, self, var_map)
