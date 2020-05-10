import numpy as np

from .. import debug
from .. import log_at_debug, log_at_info

compute_node_list = []


class INodeVisitor:
    def visit(self, computeNode):
        pass


class MComputeNode:
    def __init__(self, name=None, is_trainable=False):
        global compute_node_list
        idx = len(compute_node_list)
        compute_node_list.append(self)

        if name:
            self.name = name + "-" + str(idx)
        else:
            self.name = "$Node-" + str(idx)
        debug("Created Node named:{}".format(self.name))
        self.upstream_nodes = {}
        self.downstream_nodes = {}
        self.grad_from_downstream = {}
        self.fwd_count = 0
        self.back_count = 0
        self.node_value = None
        self._grad_value = None
        self.is_trainable = is_trainable

    def _do_check_input(self):
        pass

    def grad_value(self):
        return self._grad_value

    def reset_fwd(self):
        debug("resetting {}".format(self.simple_name()))
        self.fwd_count = 0
        self.node_value = None

    def reset_network_fwd(self):
        self.reset_fwd()
        for node in self.downstream_nodes.values():
            node.reset_network_fwd()

    def reset_back(self):
        self._grad_value = None
        self.back_count = 0
        self.grad_from_downstream = {}

    def reset_network_back(self):
        self.reset_back()
        for node in self.upstream_nodes.values():
            node.reset_network_back()

    def optimizer_step(self, optimizer, var_map):
        if self.is_trainable:
            self._optimizer_step(optimizer, var_map)
        for node in self.upstream_nodes.values():
            node.optimizer_step(optimizer, var_map)

    def _optimizer_step(self, optimizer, var_map):
        raise Exception("No implemented. Subclass responsibility")

    def _move_impl(self, optimizer, var_map):
        raise Exception("No implemented. Subclass responsibility")

    def can_go_fwd(self):
        num_upstream_nodes = len(self.upstream_nodes)
        if self.fwd_count == num_upstream_nodes:
            return True
        if self.fwd_count > len(self.upstream_nodes):
            raise Exception("Cannot be greater than number of "
                            "downstream nodes ({})".format(num_upstream_nodes))

    def __can_go_fwd(self):
        raise Exception("Not implemented. Subclass responsibility")

    def _process_backprop(self, downstream_grad, downstream_calling_node, var_values_dict):
        self.grad_from_downstream[downstream_calling_node] = downstream_grad
        self.back_count += 1
        return self.can_go_back()

    def backward(self, downstream_grad, downstream_node, var_map, tab=""):

        calling_node_name = downstream_node.simple_name()
        _value = self.value(var_map)
        if type(downstream_grad).__module__ == np.__name__:
            grad_shape = downstream_grad.shape
        else:
            grad_shape = "(float)"
        debug("{}Backprop@{} from:{} downstream grad shape:{}, value shape:{}".format(tab, self.simple_name(),
                                                                                      calling_node_name,
                                                                                      grad_shape,
                                                                                      _value.shape
                                                                                      ))
        debug(tab + "Downstream grad received:")
        debug(repr(downstream_grad))
        debug(tab + "Value:")
        debug(repr(_value))

        should_continue = self._process_backprop(downstream_grad, downstream_node, var_map)
        if not should_continue:
            return
        self._collect_grads()
        self._backprop_impl(downstream_grad, downstream_node, var_map, tab)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        raise Exception("Not implemented. Subclass responsibility")

    def _collect_grads(self):
        self._grad_value = None
        for grad_value in self.grad_from_downstream.values():
            if not self._grad_value:
                self._grad_value = np.zeros_like(grad_value)
            self._grad_value += grad_value

    def can_go_back(self):
        return self.back_count >= len(self.downstream_nodes)

    def _add_upstream_nodes(self, node_list):
        for node in node_list:
            name = node.simple_name()
            if name in self.upstream_nodes:
                raise Exception("Upstream node <{}> added multiple times".format(name))
            self.upstream_nodes[name] = node
            node.__add_downstream_node(self)

    def __add_downstream_node(self, downstream_node):
        r"""
        Must only be called by the framework ..
        :param downstream_node:
        :return:
        """
        name = downstream_node.simple_name()
        if name in self.downstream_nodes:
            raise Exception("Downstream node <{}> added multiple times".format(name))
        self.downstream_nodes[name] = downstream_node

    def _forward_downstream(self, my_value, var_map):
        for node in self.downstream_nodes.values():
            node.forward(my_value, self, var_map)

    def simple_name(self):
        return self.name

    def forward(self, var_map, upstream_value, upstream_node):
        raise Exception("Not implemented. Subclass responsibility")

    def value(self, var_map):
        r"""
        Must return last computed value or None
        :return:
        """
        return self.node_value

    def accept(self, visitor):
        visitor.visit(self)
        for upstream_node in self.upstream_nodes.values():
            upstream_node.accept(visitor)


class BinaryMatrixOp(MComputeNode):
    def __init__(self, a_node, b_node, name=None):
        MComputeNode.__init__(self, name)
        self.a_node = a_node
        self.b_node = b_node
        self._do_check_input()
        self._add_upstream_nodes([a_node, b_node])

    def forward(self, var_map, upstream_value, upstream_node):
        self.fwd_count += 1
        if not self.can_go_fwd():
            return
        self.node_value = self._do_compute(var_map)
        self._forward_downstream(self.node_value, var_map)

    def _do_compute(self, var_map):
        r"""
        implementations must return the computed value
        :param var_map:
        :return:
        """
        raise Exception("Not implemented. Subclass responsibility")


class MatrixMultiplication(BinaryMatrixOp):
    r"""
    Linear transform from R^n space to R^m
    Represents Wx
    """

    def __init__(self, a_node, b_node, name=None):
        BinaryMatrixOp.__init__(self, a_node, b_node, name)

    def _do_compute(self, var_map):
        a_matrix = self.a_node.value(var_map)
        b_matrix = self.b_node.value(var_map)
        # info("a_matrix shape:{}, b_matrix_shape:{}".format(a_matrix.shape, b_matrix.shape))
        return a_matrix @ b_matrix

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        w = self.a_node.value(var_map)
        x = self.b_node.value(var_map)
        w_grad = self._grad_value @ x.T
        x_grad = w.T @ self._grad_value
        self.a_node.backward(w_grad, self, var_map, tab + " ")
        self.b_node.backward(x_grad, self, var_map, tab + " ")


class DenseLayer(MComputeNode):
    r"""
    A direct implementation of dense layer without the full compute graph
    Designed to represent Wx +b  where W is output_dim x N and X is N X 1
    and b is output_dim x 1
    """

    def __init__(self, input_node, output_dim, initial_w=None, initial_b=None, name=None):
        r"""

        :param input_node: source of "x" also used to determine "N" of the weight matrix
        :param output_dim: dimensionality of the output vector
        :param initial_w: for custom initialization, testing, persistence etc.
        :param initial_b: for custom initialization, testing, persistence etc.
        :param name: easy to track name. This is appended by an ID to make sure names are unique
        """
        MComputeNode.__init__(self, name, is_trainable=True)
        self._add_upstream_nodes([input_node])
        self.input_node = input_node
        self.output_dim = output_dim
        self.w = initial_w
        self.b = initial_b
        self.w_grad = None
        self.b_grad = None
        self.weights_initialized = not ((self.w is None) or (self.b is None))

    def init_weights(self, input_dim):
        r"""

        :param var_map:
        :param input_dim:  number of rows in the input - dimensionality of the input
        vector
        :return:
        """
        self.w = np.random.rand(self.output_dim, input_dim)
        self.b = np.random.rand(self.output_dim).reshape((self.output_dim, 1))
        self.weights_initialized = True

    def forward(self, var_map, upstream_value, upstream_node):
        x = self.input_node.value(var_map)
        if not self.weights_initialized:
            self.init_weights(x.shape[0])
        debug("W=np.{}".format(repr(self.w)))
        debug("b=np.{}".format(repr(self.b)))
        debug("x=np.{}".format(repr(x)))
        self.node_value = self.w @ x + self.b
        self._forward_downstream(self.node_value, var_map)

    def get_component_grads(self):
        r"""

        :return: a dictionary  with 'w' containing the w gradient and 'b' containing b gradient
        """
        return {'w': self.w_grad, 'b': self.b_grad}

    def get_w_grad(self):
        return self.w_grad

    def get_b_grad(self):
        return self.b_grad

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        x = self.input_node.value(var_map)
        incoming_grad = self.grad_value()
        self.b_grad = np.average(incoming_grad, axis=1).reshape((self.output_dim, 1))
        self.b_grad = self.b_grad/incoming_grad.shape[1]
        self.w_grad = (incoming_grad @ x.T)/incoming_grad.size
        input_grad = self.w.T @ incoming_grad
        self.input_node.backward(input_grad, self, var_map, tab + " ")

    def _optimizer_step(self, optimizer, var_map):
        self.w = optimizer(self.w, self.w_grad)
        self.b = optimizer(self.b, self.b_grad)


class MatrixAddition(BinaryMatrixOp):

    def __init__(self, a_node, b_node, name=None):
        BinaryMatrixOp.__init__(self, a_node, b_node, name)

    def _do_compute(self, var_map):
        a_matrix = self.a_node.value(var_map)
        b_matrix = self.b_node.value(var_map)
        return a_matrix + b_matrix

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        a = self.a_node.value(var_map)
        b = self.b_node.value(var_map)
        a_eyes = np.ones_like(a)
        b_eyes = np.ones_like(b)
        grad_2_a = np.multiply(a_eyes, downstream_grad)
        grad_2_b = np.multiply(b_eyes, downstream_grad)
        self.a_node.backward(grad_2_a, self, var_map, tab + " ")
        self.b_node.backward(grad_2_b, self, var_map, tab + " ")


class MatrixSubtraction(BinaryMatrixOp):

    def __init__(self, a_node, b_node, name=None):
        BinaryMatrixOp.__init__(self, a_node, b_node, name)

    def _do_compute(self, var_map):
        a_matrix = self.a_node.value(var_map)
        b_matrix = self.b_node.value(var_map)
        return a_matrix - b_matrix

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        a = self.a_node.value(var_map)
        b = self.b_node.value(var_map)
        a_eyes = np.ones_like(a)
        b_eyes = np.ones_like(b)
        grad_2_a = np.multiply(a_eyes, downstream_grad)
        grad_2_b = np.multiply(b_eyes, downstream_grad)
        self.a_node.backward(grad_2_a, self, var_map, tab + " ")
        self.b_node.backward(-grad_2_b, self, var_map, tab + " ")


class VarNode(MComputeNode):
    def __init__(self, var_name, is_trainable=False):
        MComputeNode.__init__(self, var_name)
        self.var_name = var_name
        self.is_trainable = is_trainable

    def forward(self, var_map, upstream_value, upstream_node):
        self.node_value = var_map[self.var_name]
        self._forward_downstream(self.node_value, var_map)

    def _optimizer_step(self, optimizer, var_map):
        var = var_map[self.var_name]
        new_var = optimizer(var, self.grad_value())
        var_map[self.var_name] = new_var

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        pass


class SigmoidNode(MComputeNode):
    def __init__(self, upstream_node, name=None):
        MComputeNode.__init__(self, name, False)
        self.node = upstream_node
        self._add_upstream_nodes([upstream_node])

    def forward(self, var_map, upstream_value, upstream_node):
        matrix = self.node.value(var_map)
        self.node_value = 1 / (1 + np.exp(-matrix))
        self._forward_downstream(self.node_value, var_map)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        sig_grad = self.node_value * (1 - self.node_value)
        grad_downstream = sig_grad * self.grad_value()
        self.node.backward(grad_downstream, self, var_map, tab + " ")


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
        y_del = 2*(y_pred - y_act)
        y_pred_grad = y_del * self._grad_value
        y_act_grad = -y_del * self._grad_value
        self.a_node.backward(y_pred_grad, self, var_map, tab + " ")
        self.b_node.backward(y_act_grad, self, var_map, tab + " ")


def default_optimizer_function(w, grad, lr=0.01):
    return w - lr * grad


class OptimizerIterator:
    def __init__(self, start_nodes, end_node_with_loss, optimizer_function=default_optimizer_function):
        self.optimizer_function = optimizer_function
        self.start_nodes = start_nodes
        self.end_node = end_node_with_loss

    def step(self, var_map, incoming_grad):
        r"""
        Will reset the network, do forward and backward prop and then update gradient.
        The loss returned is before the gradient update
        :param var_map:
        :param incoming_grad: Starting grad, typically ones
        :return: loss before the reset and gradient updates.
        """
        for node in self.start_nodes:
            node.reset_network_fwd()
        self.end_node.reset_network_back()
        for node in self.start_nodes:
            node.forward(var_map, None, self)
        self.end_node.backward(incoming_grad, self, var_map, " ")
        loss = self.end_node.value(var_map)
        self.end_node.optimizer_step(self.optimizer_function, var_map)
        return loss

    @staticmethod
    def set_log_to_info():
        log_at_info()

    @staticmethod
    def set_log_to_debug():
        log_at_debug()

    def simple_name(self):
        return OptimizerIterator.__name__
