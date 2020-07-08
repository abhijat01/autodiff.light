import numpy as np
import core.np.Nodes as node


class BatchNormalization(node.MComputeNode):
    def __init__(self, input_node: node.MComputeNode, name=None):
        node.MComputeNode.__init__(self, name, is_trainable=True)
        self.input_node: node.MComputeNode = input_node
        self._add_upstream_nodes([input_node])
        self.optimization_storage = {'gamma': {}, 'beta': {}}
        self.mu = None
        self.sigma_square = None
        self.sig_sqrt_inv = None
        self.x_norm = None
        self.gamma = None
        self.beta = None
        self.epsilon = 1e-5
        self.beta_grad = None
        self.gamma_grad = None

    def clear_optimization_storage(self):
        self.optimization_storage = {'gamma': {}, 'beta': {}}
        for up_node in self.upstream_nodes.values():
            up_node.clear_optimization_storage()

    def forward(self, var_map):
        input_value = self.input_node.value()
        dim, m = input_value.shape
        if self.gamma is None:
            self.gamma = np.ones((dim, 1))
            self.beta = np.zeros((dim, 1))
        self.mu = np.mean(input_value, axis=1).reshape(dim, 1)
        del_mu = input_value - self.mu
        del_mu_square = np.square(del_mu)
        self.sigma_square = np.sum(del_mu_square, axis=1).reshape(dim, 1) / m
        self.sig_sqrt_inv = 1 / np.sqrt(self.sigma_square + self.epsilon)
        self.x_norm = del_mu * self.sig_sqrt_inv
        self.node_value = self.gamma * self.x_norm + self.beta
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        dim, batch_size = self._total_incoming_grad_value.shape
        self.beta_grad = np.sum(self._total_incoming_grad_value, axis=1).reshape(dim, 1)
        self.gamma_grad = self._total_incoming_grad_value * self.x_norm
        self.gamma_grad = np.sum(self.gamma_grad, axis=1).reshape(dim, 1)
        dxhat = self.node_value * self.gamma
        p1 = batch_size * dxhat
        p2 = np.sum(dxhat, axis=1).reshape(dim, 1)
        # p3 = np.sum(dxhat*self.x_norm, axis=1)
        p3 = self.x_norm * dxhat
        p3 = np.sum(p3, axis=1).reshape(dim, 1)
        p4 = self.x_norm * p3
        grad_to_input = (1. / batch_size) * self.sig_sqrt_inv * (p1 - p2 - p4)
        # grad_to_input = (1. / m) * self.sig_sqrt_inv * (m * dxhat -
        #                                                 np.sum(dxhat, axis=1) -
        #                                                  self.x_norm * np.sum(dxhat * self.x_norm, axis=1))
        self.input_node.backward(grad_to_input, self, var_map)

    def _optimizer_step(self, optimizer, var_map):
        self.beta = optimizer(self.beta, self.beta_grad, self.optimization_storage['beta'])
        self.gamma = optimizer(self.gamma, self.gamma_grad, self.optimization_storage['gamma'])


class Dropout(node.MComputeNode):
    def __init__(self, input_node: node.MComputeNode, dropout_prob: float = .5, name=""):
        node.MComputeNode.__init__(self, name)
        self.input_node = input_node
        self._add_upstream_nodes([input_node])
        self.p = dropout_prob
        self.drop_out_rows = None

    def forward(self, var_map: node.ComputeContext):
        x = self.input_node.value()
        if self.drop_out_rows is None:
            self.drop_out_rows = np.ones_like(x)
        if not var_map.is_training():
            self.node_value = x * self.p
        else:
            dim, batch_count = x.shape
            self.drop_out_rows = np.array([[np.random.binomial(1, self.p) for _ in range(dim)]]).T
            self.node_value = x * self.drop_out_rows
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        grad_to_input = self.total_incoming_gradient() * self.drop_out_rows
        self.input_node.backward(grad_to_input, self, var_map)


def batchnorm_backward(dout, cache):
    batch_size, dim = dout.shape
    x_mu, inv_var, x_hat, gamma = cache

    # intermediate partial derivatives
    dxhat = dout * gamma

    # final partial derivatives
    dx = (1. / batch_size) * inv_var * (batch_size * dxhat - np.sum(dxhat, axis=0)
                                        - x_hat * np.sum(dxhat * x_hat, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)

    return dx, dgamma, dbeta
