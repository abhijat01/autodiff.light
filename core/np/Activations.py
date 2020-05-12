import numpy as np
from core.np.Nodes import MComputeNode


class SigmoidNode(MComputeNode):
    def __init__(self, upstream_node, name=None):
        MComputeNode.__init__(self, name, False)
        self.input_node = upstream_node
        self._add_upstream_nodes([upstream_node])

    def forward(self, var_map, upstream_value, upstream_node):
        matrix = self.input_node.value(var_map)
        self.node_value = 1 / (1 + np.exp(-matrix))
        self._forward_downstream(self.node_value, var_map)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        sig_grad = self.node_value * (1 - self.node_value)
        grad_to_input = sig_grad * self.grad_value()
        self.input_node.backward(grad_to_input, self, var_map, tab + " ")


class RelUNode(MComputeNode):
    def __init__(self, upstream_node, name=None):
        MComputeNode.__init__(self, name, False)
        self.input_node = upstream_node
        self._add_upstream_nodes([upstream_node])

    def forward(self, var_map, upstream_value, upstream_node):
        input_value = self.input_node.value(var_map)
        self.node_value = input_value * (input_value > 0)
        self._forward_downstream(self.node_value, var_map)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        grad_to_input = np.ones_like(self.node_value)
        input_value = self.input_node.value(var_map)
        grad_to_input = grad_to_input * (input_value > 0)
        grad_to_input = self._grad_value*grad_to_input
        self.input_node.backward(grad_to_input, self, var_map, tab+" ")
