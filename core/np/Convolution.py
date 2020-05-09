from . import Nodes as node
import numpy as np
from core import debug


class Convolution2D(node.MComputeNode):
    r"""
    Lets start with simple square kernels and stride of 1
    """

    def __init__(self, input_node, input_shape, kern_size=2, kernel=None, bias=0, name=None):
        r"""
        if
        :param input_node:
        :param input_shape: required. Should be the expected image size
        :param kern_size:  ignored if kernel is provided.
        :param kernel:
        :param name:
        """
        node.MComputeNode.__init__(self, name, is_trainable=True)
        self.input_node = input_node
        self._add_upstream_nodes([input_node])
        self.m, self.n = input_shape
        self.size = self.m * self.n
        self.kernel_size = kern_size
        self.n_stride = 1
        self.m_stride = 1
        self.bias = bias
        if not (kernel is None):
            if not (kernel.shape[0] == kernel.shape[1]):
                raise Exception("Only square kernels supported at present. Size provided:{}".format(kernel.shape))

            self.kernel = kernel
            self.kernel_size = kernel.shape[0]
        else:
            self.kernel = np.random.rand(self.kernel_size, self.kernel_size)

    def forward(self, var_map, upstream_value, upstream_node):
        x = self.input_node.value(var_map)
        if not x.size == self.size:
            raise Exception("Expecting size:({},{}). Received:{}".format(self.m, self.n, x.shape))
        expected_shape = (self.m, self.n)

        if not expected_shape == x.shape:
            x = x.reshape(expected_shape)

        self.node_value = np.zeros((self.m - self.kernel.shape[0]+1, self.n - self.kernel.shape[1]+1))
        for i in range(self.m - self.kernel.shape[0]+1):
            i_end = i + self.kernel_size
            for j in range(self.n - self.kernel.shape[1]+1):
                j_end = j + self.kernel_size
                x_part = x[i:i_end, j:j_end]
                prod = x_part * self.kernel
                self.node_value[i, j] = np.sum(prod) + self.bias
                # debug("(i,j)=({},{})".format(i, j))
        self._forward_downstream(self.node_value, var_map)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        r"""
        Variable naming assumes x as incoming value and y as the output of the
        convolution
        :param downstream_grad:
        :param downstream_node:
        :param var_map:
        :param tab:
        :return:
        """
        x = self.input_node.value(var_map)
        x = x.reshape((self.m, self.n))
        self.x_grad = np.zeros((self.m, self.n))
        self.kernel_grad = np.zeros((self.kernel_size, self.kernel_size))
        self.bias_grad = 0

        for i in range(self.m - self.kernel.shape[0]+1):
            for j in range(self.n - self.kernel.shape[1]+1):
                y_grad_part = self._grad_value[i, j]
                self._collect_component_grads(i, j, x, y_grad_part)

        self.input_node.backward(self.x_grad, self, var_map, tab + " ")

    def _collect_component_grads(self, i, j, x, y_grad):
        x_window = x[i:i + self.kernel_size, j:j + self.kernel_size]
        grad_to_x_window = self.kernel * y_grad
        self.x_grad[i:i + self.kernel_size, j:j + self.kernel_size] += grad_to_x_window
        self.bias_grad += y_grad
        grad_to_kernel = x_window * y_grad
        self.kernel_grad += grad_to_kernel

    def get_component_grads(self):
        return {'k': self.kernel_grad, 'b': self.bias}

    def get_kernel(self):
        return self.kernel

    def get_bias(self):
        return self.bias

    def _optimizer_step(self, optimizer, var_map):
        self.kernel = optimizer(self.kernel, self.kernel_grad)
        self.bias = optimizer(self.bias, self.bias_grad)


class MaxPool2D(node.MComputeNode):
    r"""
    Simple max pool with 1 stride only
    """
    def __init__(self, input_node,  pool_size=(2, 2), name=None):
        r"""

        :param input_node:  must be  a 2D array
        :param pool_size:
        :param name:
        """
        node.MComputeNode.__init__(name)
        self.pool_size = pool_size
        self.input_node = input_node
        self.x_grad = None

    def forward(self, var_map, upstream_value, upstream_node):
        del_m = self.pool_size[0]
        del_n = self.pool_size[1]
        x = self.input_node.value(var_map)
        x_m = x.shape[0]
        x_n = x.shape[1]
        self.node_value = np.zeros((x_m-del_m+1, x_n-del_n+1))
        for i in range(x_m-del_m+1):
            for j in range(x_n-del_n+1):
                max_i, max_j = self.get_max_idx_in_window(i, j, x)
                self.node_value[i, j] = x[max_i, max_j]
        self._forward_downstream(self.node_value, var_map)

    def _backprop_impl(self, downstream_grad, downstream_node, var_map, tab=""):
        del_m = self.pool_size[0]
        del_n = self.pool_size[1]
        x = self.input_node.value(var_map)
        x_m = x.shape[0]
        x_n = x.shape[1]
        self.x_grad = np.zeros_like(x)

        for i in range(x_m-del_m+1):
            for j in range(x_n-del_n+1):
                max_i, max_j = self.get_max_idx_in_window(i, j, x)
                self.x_grad[max_i, max_j] += self._grad_value[max_i, max_j]

    def get_max_idx_in_window(self, i_start, j_start, x):
        max_i = i_start
        max_j = j_start
        max_value = None
        for i in range(i_start, i_start+self.pool_size[0]):
            for j in range(j_start, j_start+self.pool_size[1]):
                v = x[i, j]
                if not max_value:
                    max_value = v
                    max_i, max_j = i, j
                elif v > max_value:
                    max_value = v
                    max_i, max_j = i, j
        return max_i, max_j





