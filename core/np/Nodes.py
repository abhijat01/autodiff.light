r"""

This module contains many of the building blocks, in particular the common node
class, propagation code, dense layer etc.
"""


import numpy as np
from core import debug, is_debug_on, info
import math

compute_node_list = []


class INodeVisitor:
    def visit(self, computeNode):
        pass


class Initializer:
    def initialize(self, node: 'MComputeNode'):
        r"""
        Used to initializer a layer possibly using information from the node
        object.
        :param node:
        :return:
        """
        raise Exception("Subclass responsibility")


class DefaultDenseLayerWeightInitializer(Initializer):
    def initialize(self, node: 'MComputeNode'):
        if not issubclass(node.__class__, DenseLayer):
            raise Exception("Only dense layer supported at present")

        input_dim, output_dim = node.get_dimension()
        w, b = self.get_weights(input_dim, output_dim)
        node.init_params(weight=w, bias=b)

    def get_weights(self, input_dim: int, output_dim: int):
        w = np.random.rand(output_dim, input_dim)
        b = np.random.rand(output_dim).reshape((output_dim, 1))
        return w, b


class XavierInitializer(DefaultDenseLayerWeightInitializer):
    def get_weights(self, input_dim: int, output_dim: int):
        xavier_scale = math.sqrt(6.0 / (output_dim + input_dim))
        w = np.random.uniform(-1, 1, (output_dim, input_dim)) * xavier_scale
        b = np.random.rand(output_dim).reshape(output_dim, 1)
        return w, b


class ComputeContext(dict):
    def __init__(self, initial_dict={}, weight_initializer='xavier', is_training=True):
        for key, value in initial_dict.items():
            self[key] = value
        self.weight_init = weight_initializer
        self.initializers = {}
        dense_layer_name = DenseLayer.__name__
        if self.weight_init == 'xavier':
            self.initializers[dense_layer_name] = XavierInitializer()
        else:
            self.initializers[dense_layer_name] = DefaultDenseLayerWeightInitializer()
        self.training = is_training

    def set_dense_layer_initializer(self, initializer: Initializer):
        self.initializers[DenseLayer.__name__] = initializer

    def initialize_layer(self, node: 'MComputeNode'):
        class_name = node.__class__.__name__
        if not (class_name in self.initializers):
            info("[ComputeContext.initialize_layer()]  "
                 "No initializer for: {}".format(class_name))
            return
        initializer = self.initializers[class_name]
        initializer.initialize(node)

    def set_is_training(self, is_training):
        self.training = is_training

    def is_training(self):
        return self.training


class MComputeNode:
    def __init__(self, name=None, is_trainable=False):
        global compute_node_list
        idx = len(compute_node_list)
        compute_node_list.append(self)

        if name:
            self.name = name + "-" + str(idx)
        else:
            self.name = self.__class__.__name__ + str(idx)
        debug("Created Node named:{}".format(self.name))
        self.upstream_nodes = {}
        self.downstream_nodes = {}
        self.grad_from_downstream = []
        self.fwd_count = 0
        self.back_count = 0
        self.node_value = None
        self._grad_value = None
        self.is_trainable = is_trainable
        self.optimization_storage = {}

    def clear_optimization_storage(self):
        self.optimization_storage = {}
        for node in self.upstream_nodes.values():
            node.clear_optimization_storage()

    def _reset_fwd(self):
        debug("resetting {}".format(self.simple_name()))
        self.fwd_count = 0
        self.node_value = None

    def reset_network_fwd(self):
        self._reset_fwd()
        for node in self.downstream_nodes.values():
            node.reset_network_fwd()

    def _reset_back(self):
        self._grad_value = None
        self.back_count = 0
        self.grad_from_downstream = []

    def reset_network_back(self):
        self._reset_back()
        for node in self.upstream_nodes.values():
            node.reset_network_back()

    def can_go_fwd(self):
        num_upstream_nodes = len(self.upstream_nodes)
        if self.fwd_count == num_upstream_nodes:
            return True
        if self.fwd_count > len(self.upstream_nodes):
            raise Exception("Cannot be greater than number of "
                            "downstream nodes ({})".format(num_upstream_nodes))

    def backward(self, downstream_grad, downstream_node, var_map):
        r"""
        This implements some of the common processing needed for backprop to work properly.
        :param downstream_grad:   gradient coming from downstream. probably not needed. We should make
        sure that this is achieved through calling grad_value() on the node
        :param downstream_node: downstream node that invoked backward
        :param var_map:
        :return:
        """
        calling_node_name = downstream_node.simple_name()
        _value = self.value()
        if type(downstream_grad).__module__ == np.__name__:
            grad_shape = downstream_grad.shape
        else:
            grad_shape = "(float)"
        if is_debug_on():
            debug("Backprop@{} from:{} downstream grad shape:{}, value shape:{}".format(self.simple_name(),
                                                                                        calling_node_name,
                                                                                        grad_shape,
                                                                                        _value.shape
                                                                                        ))
            debug("Downstream grad received:")
            debug(repr(downstream_grad))
            debug("Value:")
            debug(repr(_value))

        should_continue = self._process_backprop(downstream_grad)
        if not should_continue:
            return
        self._collect_grads()
        self._do_backprop(downstream_grad, downstream_node, var_map)

    def _process_backprop(self, downstream_grad):
        self.grad_from_downstream.append(downstream_grad)
        self.back_count += 1
        return self._can_go_back()

    def _can_go_back(self):
        return self.back_count >= len(self.downstream_nodes)

    def _collect_grads(self):
        self._grad_value = None
        for grad_value in self.grad_from_downstream:
            if self._grad_value is None:
                self._grad_value = np.zeros_like(grad_value)
            self._grad_value += grad_value

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        r"""
        Before backprop is invoked, this node has already received all contributions
        from upstream nodes and added them together.
        :param downstream_grad:
        :param downstream_node:
        :param var_map:
        :return:
        """
        raise Exception("Not implemented. Subclass responsibility")

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

    def _forward_downstream(self, var_map):
        r"""
        Node should invoke forward after it is done processing ..
        :param var_map:
        :return:
        """
        for node in self.downstream_nodes.values():
            # node.forward(my_value, self, var_map)
            node.forward(var_map)

    def forward(self, var_map):
        raise Exception("Not implemented. Subclass responsibility")

    def value(self):
        r"""
        Must return last computed value or None
        :return:
        """
        return self.node_value

    def total_incoming_gradient(self):
        return self._grad_value

    def simple_name(self):
        return self.name

    def accept(self, visitor):
        visitor.visit(self)
        for upstream_node in self.upstream_nodes.values():
            upstream_node.accept(visitor)

    def optimizer_step(self, optimizer, var_map):
        if self.is_trainable:
            self._optimizer_step(optimizer, var_map)
        for node in self.upstream_nodes.values():
            node.optimizer_step(optimizer, var_map)

    def _optimizer_step(self, optimizer, var_map):
        raise Exception("No implemented. Subclass responsibility")

    def __repr__(self):
        return "[{}::{}]".format(self.name, self.__class__.__name__)


class BinaryMatrixOp(MComputeNode):
    def __init__(self, a_node, b_node, name=None):
        MComputeNode.__init__(self, name, is_trainable=False)
        self.a_node = a_node
        self.b_node = b_node
        self._add_upstream_nodes([a_node, b_node])

    def forward(self, var_map):
        self.fwd_count += 1
        if not self.can_go_fwd():
            return
        self.node_value = self._do_compute(var_map)
        self._forward_downstream(var_map)

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
        r"""
        value is a_node@b_node . Think of a_node as w and b_node as x
        :param a_node:
        :param b_node:
        :param name:
        """
        BinaryMatrixOp.__init__(self, a_node, b_node, name)

    def _do_compute(self, var_map):
        a_matrix = self.a_node.value()
        b_matrix = self.b_node.value()
        return a_matrix @ b_matrix

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        w = self.a_node.value()
        x = self.b_node.value()
        w_grad = self._grad_value @ x.T
        x_grad = w.T @ self._grad_value
        self.a_node.backward(w_grad, self, var_map)
        self.b_node.backward(x_grad, self, var_map)


class DenseLayer(MComputeNode):
    r"""
    A direct implementation of dense layer without the full compute graph
    Designed to represent $Wx +b$  where W is output_dim x input_dim, X is input_dim X batch_size
    and b is output_dim x 1
    """

    def __init__(self, input_node: int, output_dim: int,
                 initial_w: np.array = None, initial_b: np.array = None,
                 name: str = None):
        r"""
        Note that since the input dimension is determined at the first invocation of
        forward, this object does not yet know the input dimension and hecne cannot be
        initialized.

        :param input_node: source of "x" also used to determine "N" of the weight matrix
        :param output_dim: dimensionality of the output vector
        :param initial_w: for custom initialization, testing, persistence etc.
        :param initial_b: for custom initialization, testing, persistence etc.
        :param name: easy to track name. This is appended by an ID to make sure names are unique
        :param weight_scale: layer weight will be scaled (multiplied) by the this factor at the initialization time
        """
        MComputeNode.__init__(self, name, is_trainable=True)
        self._add_upstream_nodes([input_node])
        self.input_node = input_node
        self.output_dim = output_dim
        self.input_dim = None
        self.w = initial_w
        self.b = initial_b
        self.w_grad = None
        self.b_grad = None
        self.weights_initialized = not ((self.w is None) and (self.b is None))
        self.optimization_storage = {'w': {}, 'b': {}}

    def forward(self, var_map: ComputeContext):
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

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        x = self.input_node.value()
        incoming_grad = self.total_incoming_gradient()
        self.b_grad = np.sum(incoming_grad, axis=1).reshape((self.output_dim, 1))
        self.w_grad = (incoming_grad @ x.T)
        grad_to_input = self.w.T @ incoming_grad
        self.input_node.backward(grad_to_input, self, var_map)

    def init_params(self, weight=None, bias=None):
        self.w = weight
        self.b = bias
        self.weights_initialized = True

    def _optimizer_step(self, optimizer, var_map):
        self.w = optimizer(self.w, self.w_grad, self.optimization_storage['w'])
        self.b = optimizer(self.b, self.b_grad, self.optimization_storage['b'])

    def clear_optimization_storage(self):
        self.optimization_storage = {'w': {}, 'b': {}}
        for node in self.upstream_nodes.values():
            node.clear_optimization_storage()

    def get_w_grad(self):
        return self.w_grad

    def get_b_grad(self):
        return self.b_grad

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def get_dimension(self):
        r"""

        :return: will return tuple (input_dim, output_dim)
        """
        return self.input_dim, self.output_dim


class MatrixAddition(BinaryMatrixOp):

    def __init__(self, a_node, b_node, name=None):
        BinaryMatrixOp.__init__(self, a_node, b_node, name)

    def _do_compute(self, var_map):
        a_matrix = self.a_node.value()
        b_matrix = self.b_node.value()
        return a_matrix + b_matrix

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        a = self.a_node.value()
        b = self.b_node.value()
        a_eyes = np.ones_like(a)
        b_eyes = np.ones_like(b)
        grad_2_a = np.multiply(a_eyes, downstream_grad)
        grad_2_b = np.multiply(b_eyes, downstream_grad)
        self.a_node.backward(grad_2_a, self, var_map)
        self.b_node.backward(grad_2_b, self, var_map)


class MatrixSubtraction(BinaryMatrixOp):

    def __init__(self, a_node, b_node, name=None):
        BinaryMatrixOp.__init__(self, a_node, b_node, name)

    def _do_compute(self, var_map):
        a_matrix = self.a_node.value()
        b_matrix = self.b_node.value()
        return a_matrix - b_matrix

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        a = self.a_node.value()
        b = self.b_node.value()
        a_eyes = np.ones_like(a)
        b_eyes = np.ones_like(b)
        grad_2_a = np.multiply(a_eyes, downstream_grad)
        grad_2_b = np.multiply(b_eyes, downstream_grad)
        self.a_node.backward(grad_2_a, self, var_map)
        self.b_node.backward(-grad_2_b, self, var_map)


class VarNode(MComputeNode):
    def __init__(self, var_name, is_trainable=False):
        MComputeNode.__init__(self, var_name)
        self.var_name = var_name
        self.is_trainable = is_trainable

    def forward(self, var_map):
        self.node_value = var_map[self.var_name]
        self._forward_downstream(var_map)

    def _optimizer_step(self, optimizer, var_map):
        var = var_map[self.var_name]
        new_var = optimizer(var, self.total_incoming_gradient(), self.optimization_storage)
        var_map[self.var_name] = new_var

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        pass


def make_evaluator(start_node_list, output_node):
    return SingleOutputNetworkEvaluator(start_node_list, output_node)


class SingleOutputNetworkEvaluator:
    def __init__(self, start_nodes, output_node):
        self.start_nodes = start_nodes
        self.output_node = output_node

    def __call__(self, var_map):
        return self.at(var_map)

    def at(self, var_map):
        for start_node in self.start_nodes:
            start_node.reset_network_fwd()
        for start_node in self.start_nodes:
            start_node.forward(var_map)
        return self.output_node.value()

    def simple_name(self):
        return self.__class__.__name__
