import unittest
import core.np.Nodes as node


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


if __name__ == '__main__':
    unittest.main()
