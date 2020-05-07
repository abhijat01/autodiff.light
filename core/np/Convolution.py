from . import Nodes as node
import numpy as np

class Convolution2D(node.MComputeNode):
    def __init__(self, input_node, input_shape, kern_size=2, name=None ):
        node.MComputeNode.__init__(self, name, is_trainable=True)
        self.input_node = input_node
        self._add_upstream_nodes([input_node])
        self.h, self.w = input_shape
        self.kernel_size = kern_size

    def forward(self, var_map, upstream_value, upstream_node):
        matrix = self.node.value(var_map)
        self.node_value = 1 / (1 + np.exp(-matrix))
        self._forward_downstream(self.node_value, var_map)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        sig_grad = self.node_value * (1 - self.node_value)
        grad_downstream = sig_grad * self.grad_value()
        self.node.backward(grad_downstream, self, var_map, tab + " ")

