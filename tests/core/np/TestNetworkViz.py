import unittest
import tests.core.np.TestModels as models
import core.np.NetworkViz as net
import networkx as nx

from pyvis.network import Network


class BasicNetworkViz(unittest.TestCase):
    def test_two_layer(self):
        var_map, start_nodes, end_node = models.make__two_layer_model()
        visitor = net.NodeAccumlatorVisitor(end_node)
        visitor.start()
        visitor.finish()
        print("Root name:{}".format(end_node.simple_name()))

        pyvis_graph = Network(height='900px', width='900px', directed=True, layout=True)
        pyvis_graph.from_nx(visitor.graph)
        pyvis_graph.show_buttons(filter_=['physics'])
        pyvis_graph.show('g.html')

    def test_pyvis(self):
        nxg = nx.complete_graph(10)
        G = Network()
        G.from_nx(nxg)
        G.show("g.html")




if __name__ == '__main__':
    unittest.main()
