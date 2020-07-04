"""
Starting with the simplest possible recurrent network that process batches of size 1 only.
"""
from __future__ import annotations
import numpy as np
from core import debug, is_debug_on, info

from core.np.Nodes import MComputeNode, VarNode, ComputeContext


class RnnCell(MComputeNode):

    def __init__(self, prev_node, output_dim: int,
                 hidden_dim: int, w: VarNode, w_b: VarNode,
                 u: VarNode, u_b: VarNode,
                 hidden_state: np.Array = None,
                 name: str = None):
        r"""
        Note that since the input dimension is determined at the first invocation of
        forward, this object does not yet know the input dimension and hecne cannot be
        initialized.

        :param prev_node: source of "x" also used to determine "N" of the weight matrix
        :param output_dim: dimensionality of the output vector
        :param initial_w: for custom initialization, testing, persistence etc.
        :param initial_b: for custom initialization, testing, persistence etc.
        :param name: easy to track name. This is appended by an ID to make sure names are unique
        :param weight_scale: layer weight will be scaled (multiplied) by the this factor at the initialization time
        """
        MComputeNode.__init__(self, name, is_trainable=True)
        self._add_upstream_nodes([prev_node])
        self.input_node = prev_node
        self.is_first_rnn_cell = not isinstance(prev_node, self.__class__)
        if self.is_first_rnn_cell and (hidden_state is None):
            raise Exception("No initial hidden state provided")
        self.hidden_state = hidden_state
        self.output_dim = output_dim
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.w = w
        self.u = u
        self.w_b = w_b
        self.u_b = u_b


    def forward(self, var_map: ComputeContext):
        h = self.hidden_state
        if not self.is_first_rnn_cell:
            x, h = self.input_node.value()
        else:
            x = self.input_node.value()



        x = self.input_node.value()
        if not self.weights_initialized:
            self.input_dim = x.shape[0]
            var_map.initialize_layer(self)

        if is_debug_on():
            debug("[{}] DenseLayer.forward() W=np.{}".format(self.simple_name(), repr(self.w)))
            debug("[{}] DenseLayer.forward() b=np.{}".format(self.simple_name(), repr(self.b)))
            debug("[{}] DenseLayer.forward() x=np.{}".format(self.simple_name(), repr(x)))
        self.node_value = self.w @ x + self.b
        self._forward_downstream(var_map)
