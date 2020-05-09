import math

from . import debug


compute_node_list = []

r"""
A-->B-->C 
A is start, C is end  
for B, A is upstream , C is downstream 
Idea being regular flow is up to down or start to finish 
"""


class ComputeNode:
    def __init__(self, name):
        global compute_node_list
        idx = len(compute_node_list)
        compute_node_list.append(self)

        if name:
            self.name = name + "-" + str(idx)
        else:
            self.name = "$Node-" + str(idx)

        self.downstream_nodes = {}
        self.upstream_nodes = {}
        self.back_count = 0
        self.grad_from_downstream = {}
        self._grad_value = None

    def get_simple_name(self):
        return self.name

    def add_downstream_node(self, node):
        self.downstream_nodes[node.name] = node

    def add_upstream_nodes(self, nodes):
        for node in nodes:
            self.upstream_nodes[node.name] = node
            node.add_downstream_node(self)

    def _forward_downstream(self, my_value, var_values_dict):
        for node in self.downstream_nodes.values():
            node.forward(my_value, var_values_dict)

    def should_go_back(self):
        return self.back_count >= len(self.downstream_nodes)

    def reset_back(self):
        self.back_count = 0
        self._grad_value = None
        self.grad_from_downstream = {}

    def _process_backprop(self, downstream_grad, downstream_calling_node, var_values_dict):
        self.grad_from_downstream[downstream_calling_node] = downstream_grad
        self.back_count += 1
        return self.should_go_back()

    def _collect_grads(self):
        self._grad_value = 0
        for grad_value in self.grad_from_downstream.values():
            self._grad_value += grad_value

    def forward(self, last_value, var_values_dict):
        raise Exception("Not implemented")

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        r"""
        designed with debugging, illustrating how backprop works in mind
        :param downstream_grad: gradient coming from the node after this
        :param downstream_calling_node: node that sent the gradient back (upstream)
        :param var_values_dict: value of variables
        :param tab: used for pretty printing debugging statements  .. for debugging
        :return:
        """
        raise Exception("Not implemented")

    def value(self, var_values_dict):
        raise Exception("Not implemented")

    def grad_value(self, var_values_dict):
        return self._grad_value


class BinaryOpNode(ComputeNode):
    def __init__(self, n1, n2, name):
        ComputeNode.__init__(self, name)
        self.n1 = n1
        self.n2 = n2
        self.add_upstream_nodes([n1, n2])
        self.computed_value = None
        self.fwd_count = 0

    def should_go_fwd(self):
        return self.fwd_count == 2

    def value(self, var_value_dicts):
        return self.computed_value

    def reset_fwd(self):
        self.fwd_count = 0

    def forward(self, last_value, var_values_dict):
        self.fwd_count += 1
        if not self.should_go_fwd():
            return
        self.do_compute(var_values_dict)
        self.reset_fwd()
        my_value = self.value(var_values_dict)
        self._forward_downstream(my_value, var_values_dict)


class SumNode(BinaryOpNode):

    def __init__(self, n1, n2, name=None):
        BinaryOpNode.__init__(self, n1, n2, name)
        self.computed_value = None

    def do_compute(self, var_value_dicts):
        self.computed_value = self.n1.value(var_value_dicts) + self.n2.value(var_value_dicts)

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        calling_node_name = downstream_calling_node.get_simple_name()
        debug("{}backprop@{} From:{} Value:{},  got grad:{}".format(tab, self.get_simple_name(),
                                                                    calling_node_name,
                                                                    self.value(var_values_dict),
                                                                    downstream_grad))
        should_continue = self._process_backprop(downstream_grad, downstream_calling_node, var_values_dict)
        if not should_continue:
            return
        self._collect_grads()

        self.n1.backward(self._grad_value, self, var_values_dict, tab + tab)
        self.n2.backward(self._grad_value, self, var_values_dict, tab + tab)


class ProdNode(BinaryOpNode):
    def __init__(self, n1, n2, name=None):
        BinaryOpNode.__init__(self, n1, n2, name)
        self.computed_value = None

    def do_compute(self, var_value_dicts):
        self.computed_value = self.n1.value(var_value_dicts) * self.n2.value(var_value_dicts)

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        calling_node_name = downstream_calling_node.get_simple_name()
        debug("{}backprop@{}, From:{} Value:{},  got grad:{}".format(tab, self.get_simple_name(),
                                                                     calling_node_name,
                                                                     self.value(var_values_dict),
                                                                     downstream_grad))
        should_continue = self._process_backprop(downstream_grad, downstream_calling_node, var_values_dict)
        if not should_continue:
            return
        self._collect_grads()

        n1_grad_part = self._grad_value * self.n2.value(var_values_dict)
        n2_grad_part = self._grad_value * self.n1.value(var_values_dict)
        self.n1.backward(n1_grad_part, self, var_values_dict, tab + tab)
        self.n2.backward(n2_grad_part, self, var_values_dict, tab + tab)


class DiffNode(BinaryOpNode):
    def __init__(self, n1, n2, name=None):
        BinaryOpNode.__init__(self, n1, n2, name)
        self.computed_value = None

    def do_compute(self, var_value_dicts):
        self.computed_value = self.n1.value(var_value_dicts) - self.n2.value(var_value_dicts)

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        calling_node_name = downstream_calling_node.get_simple_name()
        debug("{}backprop@{} From:{} Value:{},  got grad:{}".format(tab, self.get_simple_name(),
                                                                    calling_node_name,
                                                                    self.value(var_values_dict),
                                                                    downstream_grad))
        should_continue = self._process_backprop(downstream_grad, downstream_calling_node, var_values_dict)
        if not should_continue:
            return
        self._collect_grads()

        self.n1.backward(self._grad_value, self, var_values_dict, tab + tab)
        self.n2.backward(-self._grad_value, self, var_values_dict, tab + tab)


class LogNode(ComputeNode):
    def __init__(self, node, name=None):
        ComputeNode.__init__(self, name)
        self.node = node
        self.add_upstream_nodes([node])
        self.log = None

    def value(self, var_value_dicts):
        return self.log

    def forward(self, last_value, var_values_dict):
        self.log = math.log(last_value)
        self._forward_downstream(self.log, var_values_dict)

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        calling_node_name = downstream_calling_node.get_simple_name()
        debug("{}backprop@{}, From:{} Value:{},  got grad:{}".format(tab, self.get_simple_name(),
                                                                     calling_node_name,
                                                                     self.value(var_values_dict),
                                                                     downstream_grad))
        should_continue = self._process_backprop(downstream_grad, downstream_calling_node, var_values_dict)
        if not should_continue:
            return
        self._collect_grads()

        self._grad_value = self._grad_value / self.node.value(var_values_dict)
        self.node.backward(self._grad_value, self, var_values_dict, tab + tab)


class SinNode(ComputeNode):
    def __init__(self, node, name=None):
        ComputeNode.__init__(self, name)
        self.node = node
        self.add_upstream_nodes([node])
        self.sine = None

    def value(self, var_value_dicts):
        return self.sine

    def forward(self, last_value, var_value_dicts):
        self.sine = math.sin(last_value)
        self._forward_downstream(self.sine, var_value_dicts)

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        calling_node_name = downstream_calling_node.get_simple_name()
        debug("{}backprop@{}, From:{},  Value:{},  got grad:{}".format(tab, self.get_simple_name(), calling_node_name,
                                                                       self.value(var_values_dict), downstream_grad))
        should_continue = self._process_backprop(downstream_grad, downstream_calling_node, var_values_dict)
        if not should_continue:
            return
        self._collect_grads()

        self._grad_value = self._grad_value * math.cos(self.node.value(var_values_dict))
        self.node.backward(self._grad_value, self, var_values_dict, tab + tab)


class VarNode(ComputeNode):
    def __init__(self, var_name):
        ComputeNode.__init__(self, var_name)
        self.var_name = var_name

    def value(self, var_value_dicts):
        return var_value_dicts[self.var_name]

    def forward(self, last_value, var_values_dict):
        my_value = self.value(var_values_dict)
        self._forward_downstream(my_value, var_values_dict)

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        calling_node_name = downstream_calling_node.get_simple_name()
        debug("{}backprop@{}, From:{} var:[{}],  got grad:{}".format(tab, self.get_simple_name(), calling_node_name,
                                                                     self.var_name, downstream_grad))
        should_continue = self._process_backprop(downstream_grad, downstream_calling_node, var_values_dict)
        if not should_continue:
            return
        self._collect_grads()


class EndValueCollectorNode(ComputeNode):
    r"""
    Not sure if this is needed. Should look into removing this or replacing with a unit op node
    which will make more sense
    """
    def __init__(self, node, name="Term"):
        ComputeNode.__init__(self, name)
        self.node = node
        self.add_upstream_nodes([node])
        self.node_value = None

    def value(self, var_values_dict):
        return self.node_value

    def forward(self, last_value, var_values_dict):
        self.node_value = last_value
        # Nothing to forward now !

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        calling_node_name = downstream_calling_node.get_simple_name()
        debug("{}backprop@{}, From:{} var:[{}],  got grad:{}".format(tab, self.get_simple_name(), calling_node_name,
                                                                     self.node_value, downstream_grad))
        self.node.backward(1.0, self, var_values_dict, tab + tab)


class ConstantOpNode(ComputeNode):
    def __init__(self, constant_value, name=None):
        ComputeNode.__init__(self, name)
        self.constant_value = constant_value

    def value(self, var_value_dicts):
        return self.constant_value

    def forward(self, last_value, var_values_dict):
        my_value = self.value(var_values_dict)
        self._forward_downstream(my_value, var_values_dict)

    def backward(self, downstream_grad, downstream_calling_node, var_values_dict, tab=""):
        self._grad_value = 0
