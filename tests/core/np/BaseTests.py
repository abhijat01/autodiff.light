import unittest


class BaseComputeNodeTest(unittest.TestCase):

    def simple_name(self):
        return self.__class__.__name__

    def default_optimizer_func(self, learning_rate=0.001):
        def optimizer(w, grad):
            return w - learning_rate * grad

        return optimizer

    def rate_adjustable_optimizer_func(self, initial_rate=0.1):
        self.learning_rate = initial_rate

        def optimizer(w, grad, node_local_storage={}):
            return w - self.learning_rate*grad
        return optimizer


