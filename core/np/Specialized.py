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
        self.epsilon = 1e-10
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
            self.beta = np.random.rand(dim,m)
        self.mu = np.mean(input_value, axis=1).reshape(dim, 1)
        del_mu = input_value - self.mu
        del_mu_square = np.square(del_mu)
        self.sigma_square = np.sum(del_mu_square, axis=1).reshape(dim, 1) / m
        self.sig_sqrt_inv = 1 / np.sqrt(self.sigma_square + self.epsilon)
        self.x_norm = del_mu * self.sig_sqrt_inv
        self.node_value = self.gamma * self.x_norm + self.beta
        self._forward_downstream(var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        dim, m = self._grad_value.shape
        self.beta_grad = np.sum(self._grad_value, axis=1).reshape(dim, 1)
        self.gamma_grad = self._grad_value * self.x_norm
        self.gamma_grad = np.sum(self.gamma_grad, axis=1).reshape(dim, 1)
        dxhat = self.node_value * self.gamma
        p1 = m*dxhat
        p2 = np.sum(dxhat, axis=1).reshape(dim,1)
        #p3 = np.sum(dxhat*self.x_norm, axis=1)
        p3 = self.x_norm*dxhat
        p3 = np.sum(p3, axis=1).reshape(dim,1)
        p4 = self.x_norm*p3
        grad_to_input = (1./m) * self.sig_sqrt_inv * (p1 - p2 - p4)
        # grad_to_input = (1. / m) * self.sig_sqrt_inv * (m * dxhat -
        #                                                 np.sum(dxhat, axis=1) -
        #                                                  self.x_norm * np.sum(dxhat * self.x_norm, axis=1))
        self.input_node.backward(grad_to_input, self, var_map )


    def _optimizer_step(self, optimizer, var_map):
        self.beta = optimizer(self.beta, self.beta_grad, self.optimization_storage['beta'])
        self.gamma = optimizer(self.gamma, self.gamma_grad, self.optimization_storage['gamma'])

