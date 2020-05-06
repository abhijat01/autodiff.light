import core.np.Nodes as cnodes
import networkx as nx


class NodeAccumlatorVisitor(cnodes.INodeVisitor):
    def __init__(self, start_node):
        self.graph = nx.DiGraph()
        self.start_node = start_node

    def start(self):
        self.start_node.accept(self)

    def visit(self, compute_node):
        node_type_name = compute_node.__class__.__name__
        node_name = compute_node.simple_name()
        self.graph.add_node(node_name, node=compute_node, type_name=node_type_name, name=node_name)

    def finish(self):
        self.positions = {}
        y, xmin, xmax = 500, 10, 12000
        #self.positions[self.start_node.simple_name()] = (int((xmax + xmin) / 2.0), y)
        self.connect_upstream(self.start_node, y + 20, xmin, xmax)

    def connect_upstream(self, node, y, x1, x2):
        node_id = node.simple_name()
        count = len(node.upstream_nodes)
        x = int((x2 + x1) / 2.0)
        self.positions[node_id] = (x, y)
        del_x = float(x2 - x1) / float(count+1)
        counter = 0
        for upstream_node in node.upstream_nodes.values():
            u_node_id = upstream_node.simple_name()
            new_x1 = x1 + del_x * counter
            new_x2 = x1 + del_x * (counter + 1)
            counter += 1
            self.positions[u_node_id] = (x, y)
            self.graph.add_edge(u_node_id, node_id)
            self.connect_upstream(upstream_node, y - 20, new_x1, new_x2)
