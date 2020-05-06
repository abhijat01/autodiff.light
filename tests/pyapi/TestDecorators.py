import unittest


class FunctionInterceptDecorator(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        print("function called")



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.hello_world("-----")

    @FunctionInterceptDecorator
    def hello_world(self):
        print("hello_world")

    def test_introspection(self):
        print(self.__class__.__name__)


if __name__ == '__main__':
    unittest.main()
