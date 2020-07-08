"""
Starting with the simplest possible recurrent network that process batches of size 1 only.
"""
from __future__ import annotations
import numpy as np
from core import debug, is_debug_on, info
import math
import core.np.Activations as act
from core.np.Nodes import MComputeNode, ComputeContext


class SimpleRnnLayer(MComputeNode):
    r"""
    Assumes that dim 0 is the dimensionality of the input vector, dimension 1 is batch dimension and dimension 2
    contains the sequence. Hence x[:,:,0] is the X vector for t0 and so on. X[:,0,0] is the first input for the
    first batch
    """

    def __init__(self, prev_node: MComputeNode, output_dim: int, state_dim: int, output_all_sequences: bool = False,
                 name: str = None):
        """

        :param prev_node: A non RNN previous node. This will be changed soon to make it more flexible
        :param output_dim: dimension of "Y"
        :param state_dim: dimension of "H", the hidden state
        :param output_all_sequences: Only False is supported at present.
        :param name:
        """
        MComputeNode.__init__(self, name=name, is_trainable=True)
        self.input_node = prev_node
        self._add_upstream_nodes([prev_node])
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.output_all_sequences = output_all_sequences
        self.weights_initialized = False
        self.initial_state = np.zeros((self.state_dim, 1))

    def set_initial_state_to_zero(self):
        h0 = np.zeros((self.state_dim, 1))
        self.set_initial_state(h0)

    def set_initial_state(self, state_vector):
        r"""
        This is the h0 .
        :param state_vector:
        :return:
        """
        if not (state_vector.shape == self.initial_state.shape):
            raise Exception("Invalid state vector shape. Supplied shape:{}, expected shape:{}",
                            state_vector.shape, self.initial_state.shape)
        self.initial_state = state_vector

    def _init_weights(self, input_x_dim, compute_context: ComputeContext):
        xavier_scale = math.sqrt(6.0 / (self.output_dim + input_x_dim + self.state_dim))
        w = np.random.uniform(-1, 1, (self.output_dim, input_x_dim + self.state_dim)) * xavier_scale
        wb = np.random.rand(self.output_dim).reshape(-1, 1)
        xavier_scale = math.sqrt(6.0) / (self.state_dim + input_x_dim + self.state_dim)
        u = np.random.uniform(-1, 1, (self.state_dim, input_x_dim + self.state_dim)) * xavier_scale
        ub = np.random.rand(self.state_dim).reshape(-1, 1)
        self.w: SharedParam = SharedParam(w)
        self.wb: SharedParam = SharedParam(wb)
        self.u: SharedParam = SharedParam(u)
        self.ub: SharedParam = SharedParam(ub)
        self.weights_initialized = True

    def forward(self, var_map):
        x = self.input_node.value()
        x_dim = x.shape
        if not self.weights_initialized:
            self._init_weights(x_dim[0], var_map)
        self.time_steps = x_dim[2]
        self.rnn_cells = []
        self.rnn_vars = []
        idx = 0
        rnn_var = RnnVarNode(idx, x)
        rnn_cell = RnnCell(rnn_var, None, self.w, self.wb, self.u, self.ub, self.initial_state)

        self.rnn_vars.append(rnn_var)
        self.rnn_cells.append(rnn_cell)

        for i in range(1, self.time_steps):
            rnn_var = RnnVarNode(i, x)
            rnn_cell = RnnCell(rnn_var, rnn_cell, self.w, self.wb, self.u, self.ub)
            self.rnn_vars.append(rnn_var)
            self.rnn_cells.append(rnn_cell)

        for rnn_var in self.rnn_vars:
            rnn_var.forward(var_map)

        self.last_y, self.last_h = self.rnn_cells[-1].value()
        self.node_value = self.last_y
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        r"""
        Will work only with a single output at the end RNN. Won't work when gradients come through
        all outputs.
        :param downstream_grad:
        :param downstream_node:
        :param var_map:
        :return:
        """
        h_grad = np.zeros((self.state_dim, 1))
        y_grad = self.total_incoming_gradient()
        self.rnn_cells[-1].backward((y_grad, h_grad), downstream_node, var_map)
        x_grad = np.zeros_like(self.input_node.value())
        for i in range(self.time_steps):
            x_grad[:, :, i] = self.rnn_vars[i].total_incoming_gradient()
        self.input_node.backward(x_grad, self, var_map)

    def _optimizer_step(self, optimizer, var_map):
        self.w._optimizer_step(optimizer, var_map)
        self.wb._optimizer_step(optimizer, var_map)
        self.u._optimizer_step(optimizer, var_map)
        self.ub._optimizer_step(optimizer, var_map)


class RnnVarNode(MComputeNode):

    def __init__(self, time_idx, input_value):
        MComputeNode.__init__(self, str(time_idx), is_trainable=False)
        self.node_value_ref = input_value
        self.time_index = time_idx
        self.node_value = self.node_value_ref[:, :, self.time_index]

    def forward(self, var_map):
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        pass


class RnnCell(MComputeNode):
    """
    It should be pretty cheap to create a new instance for each iteration!
    """

    def __init__(self, x_node: MComputeNode, prev_rnn_node: RnnCell,
                 w: SharedParam, w_b: SharedParam,
                 u: SharedParam, u_b: SharedParam,
                 hidden_state_in: np.Array = None,
                 name: str = None):
        r"""
        Note that since the input dimension is determined at the first invocation of
        forward, this object does not yet know the input dimension and hecne cannot be
        initialized.
        :param x_node: source of x (input) value
        :param prev_rnn_node: used to get the previous state
        :param w: shared weight for X transformation
        :param w_b: shared bias for X
        :param u: shared weight for hidden state
        :param u_b:  shared bias for hidden state
        :param hidden_state_in: must be specified if this is the first cell in a sequence
        :param name: easy to track name. This is appended by an ID to make sure names are unique
        """
        MComputeNode.__init__(self, name, is_trainable=False)
        if prev_rnn_node is None:
            # this is the first node
            self._add_upstream_nodes([x_node])
            if hidden_state_in is None:
                raise Exception("No initial hidden state provided")
            self.hidden_state_in = hidden_state_in
        else:
            self._add_upstream_nodes([x_node, prev_rnn_node])
            self.hidden_state_in = hidden_state_in

        self.x_node = x_node
        self.prev_rnn_node = prev_rnn_node
        self.hidden_state_out = None

        self.W: SharedParam = w
        self.U: SharedParam = u
        self.Wb: SharedParam = w_b
        self.Ub: SharedParam = u_b
        self.output_shape = self.W.shape()
        self.hidden_shape = self.U.shape()

    def _hidden_state_out(self):
        return self.hidden_state_out

    def forward(self, var_map: ComputeContext):
        if self.prev_rnn_node is None:
            h0 = self.hidden_state_in
        else:
            h0 = self.prev_rnn_node._hidden_state_out()

        x = self.x_node.value()
        xh = np.concatenate((x, h0))

        w = self.W.value()
        wb = self.Wb.value()
        u = self.U.value()
        ub = self.Ub.value()

        y = w @ xh + wb
        h = u @ xh + ub

        self.node_value = y, h
        self.hidden_state_out = h

        if is_debug_on():
            debug("[{}] RnnCell.forward() y=np.{}".format(self.simple_name(), repr(y)))
            debug("[{}] RnnCell.forward() h=np.{}".format(self.simple_name(), repr(h)))
            debug("[{}] RnnCell.forward() x=np.{}".format(self.simple_name(), repr(x)))

        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        r"""

        :param downstream_grad: should be ignored.
        :param downstream_node: should be ignored
        :param var_map:
        :return:
        """
        if self.prev_rnn_node is None:
            h0 = self.hidden_state_in
        else:
            h0 = self.prev_rnn_node._hidden_state_out()

        x = self.x_node.value()
        xh = np.concatenate((x, h0))
        incoming_y_grad, incoming_h_grad = self.total_incoming_gradient()
        if incoming_y_grad is None:
            if is_debug_on():
                debug("[RnnCell._do_backprop()] null incoming_y_grad ")
            incoming_y_grad = np.zeros((self.output_shape[0], 1))
        if incoming_h_grad is None:
            if is_debug_on():
                debug("[RnnCell._do_backprop()] Null incoming_h_grad = np.{}")
            incoming_h_grad = np.zeros((self.hidden_shape[0], 1))

        w_grad = incoming_y_grad @ xh.T
        u_grad = incoming_h_grad @ xh.T
        wb_grad = np.sum(incoming_y_grad, axis=1).reshape((-1, 1))
        ub_grad = np.sum(incoming_h_grad, axis=1).reshape((-1, 1))
        x_dim = x.shape[0]
        w = self.W.value()
        u = self.U.value()
        x_grad_from_y = w[:, 0:x_dim].T @ incoming_y_grad
        x_grad_from_h = u[:, 0:x_dim].T @ incoming_h_grad
        h_grad_from_y = w[:, x_dim:].T @ incoming_y_grad
        h_grad_from_h = u[:, x_dim:].T @ incoming_h_grad
        x_total_grad = (x_grad_from_h + x_grad_from_y)
        h_total_grad = (h_grad_from_h + h_grad_from_y)

        self.W.add_gradient(w_grad, self)
        self.Wb.add_gradient(wb_grad, self)
        self.U.add_gradient(u_grad, self)
        self.Ub.add_gradient(ub_grad, self)

        self.x_node.backward(x_total_grad, self, var_map)
        if not (self.prev_rnn_node is None):
            self.prev_rnn_node.backward((None, h_total_grad), self, var_map)

    def _collect_grads(self):
        y_grad, h_grad = self.grad_from_downstream[0]
        for yg, hg in self.grad_from_downstream[1:]:
            y_grad += yg
            h_grad += hg
        self._set_total_gradient((y_grad, h_grad))


class SharedParam:
    def __init__(self, param_matrix: np.array):
        self.param = param_matrix
        self.downstream_grads = np.zeros_like(param_matrix)
        self.local_storage = {}

    def add_gradient(self, downstream_grad: np.array, node: MComputeNode):
        self.downstream_grads += downstream_grad

    def _total_incoming_gradient(self):

        incoming_grad = self.downstream_grads
        total_grad = incoming_grad[0]
        for grad in incoming_grad[1:]:
            total_grad += grad

        norm  = np.linalg.norm(total_grad,2)
        #norm = self.squared_norm(total_grad)
        if norm > 10000:
            total_grad = total_grad / norm
        return total_grad

    def squared_norm(self, x):
        return np.amax(x)
        return sum

    def value(self):
        return self.param

    def shape(self):
        return self.value().shape

    def _optimizer_step(self, optimizer, var_map):
        total_grad = self._total_incoming_gradient()
        self.param = optimizer(self.param, total_grad, self.local_storage)
