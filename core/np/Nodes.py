import numpy as np

from .. import log_debug as debug, log_info as info
from .. import log_level_debug as set_debug, log_level_info as set_info

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
        debug(tab+"Downstream grad received:")
        debug(downstream_grad)
        debug(tab+"Value:")
        debug(_value)

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
        self._do_compute(var_map)
        self._forward_downstream(self.node_value, var_map)

    def _do_compute(self, var_map):
        raise Exception("Not implemented. Subclass responsibility")


class MatrixMult(BinaryMatrixOp):
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
        self.node_value = a_matrix @ b_matrix


class LinearTransform(MatrixMult):
    r"""
    Designed to represent Wx where W is M x N and X is N X 1 . W represents the linear
    transform
    """

    def __init__(self, w_node, x_node, name=None):
        MatrixMult.__init__(self, w_node, x_node, name)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        w = self.a_node.value(var_map)
        w_shape = w.shape
        x_ones = np.ones((w_shape[0], 1))
        # info("_grad_shape:{}".format(self._grad_value.shape))
        local_grad = self._grad_value.reshape((w_shape[0], 1))
        x_ones_with_grad = np.multiply(x_ones, local_grad)

        grad_to_w = x_ones_with_grad @ self.b_node.value(var_map).T
        grad_to_x = np.multiply(w, local_grad).sum(axis=0).T
        grad_to_x = np.reshape(grad_to_x, (len(grad_to_x), 1))

        self.a_node.backward(grad_to_w, self, var_map, tab + " ")
        self.b_node.backward(grad_to_x, self, var_map, tab + " ")


class MatrixAddition(BinaryMatrixOp):

    def __init__(self, a_node, b_node, name=None):
        BinaryMatrixOp.__init__(self, a_node, b_node, name)

    def _do_compute(self, var_map):
        a_matrix = self.a_node.value(var_map)
        b_matrix = self.b_node.value(var_map)
        self.node_value = a_matrix + b_matrix

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        a_node = self.a_node
        b_node = self.b_node
        a_eyes = np.ones_like(a_node.value(var_map))
        b_eyes = np.ones_like(b_node.value(var_map))
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
        self.node_value = a_matrix - b_matrix


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
        self.node_value = np.sum(np.square(y_del))

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        y_pred = self.a_node.value(var_map)
        y_act = self.b_node.value(var_map)
        y_del = 2 * (y_pred - y_act)
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
        set_info()

    @staticmethod
    def set_log_to_debug():
        set_debug()

    def simple_name(self):
        return OptimizerIterator.__name__
