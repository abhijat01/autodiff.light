from . import Nodes as node
import numpy as np


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
        self.x_contrib_tracker = None
        self.w_contrib_tracker = None
        self.optimization_storage = {'w': {}, 'b': {}}

    def clear_optimization_storage(self):
        self.optimization_storage = {'w': {}, 'b': {}}
        for up_node in self.upstream_nodes.values():
            up_node.clear_optimization_storage()

    def forward(self, var_map):
        x = self.input_node.value()
        if not x.size == self.size:
            raise Exception("Expecting size:({},{}). Received:{}".format(self.m, self.n, x.shape))
        expected_shape = (self.m, self.n)

        if not expected_shape == x.shape:
            x = x.reshape(expected_shape)

        self.x_contrib_tracker = ConvContributionTracker(x.shape)
        self.w_contrib_tracker = ConvContributionTracker(self.kernel.shape)
        self.node_value = np.zeros((self.m - self.kernel.shape[0] + 1, self.n - self.kernel.shape[1] + 1))
        for i in range(self.m - self.kernel_size + 1):
            i_end = i + self.kernel_size
            for j in range(self.n - self.kernel_size + 1):
                j_end = j + self.kernel_size
                x_part = x[i:i_end, j:j_end]
                prod = x_part * self.kernel
                self.node_value[i, j] = np.sum(prod) + self.bias
                y_contrib = (i, j)
                w_i = 0
                for m in range(i, i_end):
                    w_j = 0
                    for n in range(j, j_end):
                        w_contrib = (w_i, w_j)
                        self.x_contrib_tracker.add_contribution(m, n, w_contrib, y_contrib)

                # debug("(i,j)=({},{})".format(i, j))
        self._forward_downstream( var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        r"""
        Variable naming assumes x as incoming value and y as the output of the
        convolution
        :param downstream_grad:
        :param downstream_node:
        :param var_map:
        :return:
        """
        x = self.input_node.value()
        x = x.reshape((self.m, self.n))
        self.gradient_to_x = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                contributions = self.x_contrib_tracker.get_contributions(i, j)
                for wy_dict in contributions:
                    wi, wj = wy_dict['w']
                    yi, yj = wy_dict['y']
                    self.gradient_to_x[i][j] += self.kernel[wi, wj] * self._grad_value[yi][yj]

        self.kernel_grad = np.zeros((self.kernel_size, self.kernel_size))
        grad_m, grad_n = self._grad_value.shape
        for i in range(self.m - grad_m + 1):
            for j in range(self.n - grad_n + 1):
                x_part = x[i:i + grad_m, j:j + grad_n] * self._grad_value
                self.kernel_grad[i, j] = np.sum(x_part) / x_part.size

        self.bias_grad = np.sum(self._grad_value) / self._grad_value.size
        self.input_node.backward(self.gradient_to_x, self, var_map)

    def _collect_component_grads(self, i, j, x, y_grad):
        x_window = x[i:i + self.kernel_size, j:j + self.kernel_size]
        grad_to_x_window = self.kernel * y_grad
        self.gradient_to_x[i:i + self.kernel_size, j:j + self.kernel_size] += grad_to_x_window
        grad_to_kernel = x_window * y_grad
        self.kernel_grad += grad_to_kernel

    def get_kernel_grad(self):
        return self.kernel_grad

    def get_bias_grad(self):
        return self.bias_grad

    def get_kernel(self):
        return self.kernel

    def get_bias(self):
        return self.bias

    def _optimizer_step(self, optimizer, var_map):
        self.kernel = optimizer(self.kernel, self.kernel_grad, self.optimization_storage['w'])
        self.bias = optimizer(self.bias, self.bias_grad, self.optimization_storage['b'])


class ConvContributionTracker:
    def __init__(self, shape):
        self.contribs = []
        for i in range(shape[0]):
            row = []
            self.contribs.append(row)
            for j in range(shape[1]):
                row.append([])

    def add_contribution(self, i, j, w_tuple, y_tuple):
        contrib = {'w': w_tuple, 'y': y_tuple}
        self.contribs[i][j].append(contrib)

    def get_contributions(self, i, j):
        return self.contribs[i][j]


class MaxPool2D(node.MComputeNode):
    r"""
    Simple max pool with 1 stride only
    """

    def __init__(self, input_node, pool_size=(2, 2), name=None):
        r"""

        :param input_node:  must be  a 2D array
        :param pool_size:
        :param name:
        """
        node.MComputeNode.__init__(self, name)
        self.input_node = input_node
        self._add_upstream_nodes([input_node])
        self.pool_size = pool_size
        self.x_grad = None

    def forward(self, var_map):
        del_m = self.pool_size[0]
        del_n = self.pool_size[1]
        x = self.input_node.value()
        x_m = x.shape[0]
        x_n = x.shape[1]
        self.node_value = np.zeros((x_m - del_m + 1, x_n - del_n + 1))
        for i in range(x_m - del_m + 1):
            for j in range(x_n - del_n + 1):
                max_i, max_j = self.get_max_idx_in_window(i, j, x)
                self.node_value[i, j] = x[max_i, max_j]
        self._forward_downstream( var_map)

    def _do_backprop(self, downstream_grad, downstream_node, var_map):
        del_m = self.pool_size[0]
        del_n = self.pool_size[1]
        x = self.input_node.value()
        x_m = x.shape[0]
        x_n = x.shape[1]
        self.x_grad = np.zeros_like(x)

        for i in range(x_m - del_m + 1):
            for j in range(x_n - del_n + 1):
                max_i, max_j = self.get_max_idx_in_window(i, j, x)
                self.x_grad[max_i, max_j] += self._grad_value[i, j]

        self.input_node.backward(self.x_grad, self, var_map)

    def get_max_idx_in_window(self, i_start, j_start, x):
        max_i = i_start
        max_j = j_start
        max_value = None
        for i in range(i_start, i_start + self.pool_size[0]):
            for j in range(j_start, j_start + self.pool_size[1]):
                v = x[i, j]
                if not max_value:
                    max_value = v
                    max_i, max_j = i, j
                elif v > max_value:
                    max_value = v
                    max_i, max_j = i, j
        return max_i, max_j

