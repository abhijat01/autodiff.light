import unittest


class BaseComputeNodeTest(unittest.TestCase):

    def simple_name(self):
        return self.__class__.__name__
