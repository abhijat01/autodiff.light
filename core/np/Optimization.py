import numpy as np
from core import debug, info

def default_optimizer_function(w, grad, node_local_storage={}):
    r"""

    :param w:
    :param grad:
    :param lr:
    :param node_local_storage: an optional dictionary that stores optimization information local to the
    node. Useful when implementing algorithms that make use of past gradient and momentum information.
    :return:
    """
    return w - 0.01 * grad


class WeightUpdater:
    def step(self, w, grad_w, node_local_store: dict):
        raise Exception("Subclass responsibility")


class SGDOptimizer(WeightUpdater):
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, w, grad_w, node_local_store: dict):
        return w - self.lr*grad_w


class AdamOptimizer(WeightUpdater):
    r"""
    Ref: ADAM: A method for stochastic optimization, Kingma, Lei Ba (2017)
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.t = 0
        self.epsilon = 1e-8

    def step(self, theta, grad_theta, node_local_store: dict):
        if not ('vt' in node_local_store):
            node_local_store['vt'] = np.zeros_like(theta)
        if not ('mt' in node_local_store):
            node_local_store['mt'] = np.zeros_like(theta)
        mt = node_local_store['mt']
        vt = node_local_store['vt']

        self.t += 1

        mt_p_1 = (self.beta1 * mt) + ((1 - self.beta1) * grad_theta)
        vt_p_1 = (self.beta2 * vt) + ((1 - self.beta2) * np.square(grad_theta))

        mt_prime = mt_p_1 / (1 - np.power(self.beta1, self.t))
        vt_prime = vt_p_1 / (1 - np.power(self.beta2, self.t))

        node_local_store['vt'] = vt_p_1
        node_local_store['mt'] = mt_p_1
        vt_sqrt = np.sqrt(vt_prime)+self.epsilon
        new_weight = theta - (self.lr * mt_prime / vt_sqrt)
        return new_weight


class OptimizerIterator:
    def __init__(self, start_nodes, end_node_with_loss, optimizer_function=default_optimizer_function):
        r"""

        :param start_nodes:
        :param end_node_with_loss:
        :param optimizer_function: either a function that takes three variables, weight, gradient of the
        weight and a dictionary used for storing intermediate values
        or it can be an instance of WeightUpdater
        """
        if isinstance(optimizer_function, WeightUpdater):
            self.optimizer_function = optimizer_function.step
        else:
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
            node.forward(var_map)
        self.end_node.backward(incoming_grad, self, var_map)
        loss = self.end_node.value()
        self.end_node.optimizer_step(self.optimizer_function, var_map)
        return loss

    def simple_name(self):
        return self.__class__.__name__
