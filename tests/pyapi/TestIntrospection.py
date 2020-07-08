import unittest
import core.np.Nodes as node

class Base:
    def __init__(self, name):
        self._name = name

    def print_name(self):
        print("[Base.print_name()] self._name = {}".format(self._name))


class Child(Base):

    def __init__(self, name):
        Base.__init__(self, name)

    def change_name(self):
        self._name = "Another name"

class MyTestCase(unittest.TestCase):
    def test_something(self):
        var  = node.VarNode('')
        dense = node.DenseLayer(var, 100)
        print(dense.__class__.__name__)

        if isinstance(dense, node.MComputeNode):
            print("Is a compute node")
        else:
            print("Is not a compute node")

        mcomputeNode = node.MComputeNode()
        class_obj = mcomputeNode.__class__
        if isinstance(dense, class_obj):
            print("OK .. is compute node")
        else:
            print("No a compute node..")

    def test_field_access(self):
        base = Child("BaseName")
        base.print_name()
        base.change_name()
        base.print_name()

        


if __name__ == '__main__':
    unittest.main()
