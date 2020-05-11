import core.np.Nodes as node


x_node = node.VarNode('x')
y_node = node.VarNode('y')
dense = node.DenseLayer(x_node, 2)
sigmoid = node.SigmoidNode(dense)
final_dense = node.DenseLayer(sigmoid, 4)
l2_node = node.L2DistanceSquaredNorm(final_dense, y_node)