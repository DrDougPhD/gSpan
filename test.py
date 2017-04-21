import pprint

import networkx
import numpy
numpy.random.seed(0)
import unittest

class RandomGraph(object):
    vertex_line_format = 'v {vid} {label}'
    edge_line_format = 'e {uid} {vid} {label}'
    available_label = 0

    def __init__(self, num_nodes, num_edges, node_label_generator,
                 edge_label_generator):
        self._assert_legal_parameters(num_nodes, num_edges)

        G = networkx.Graph(label=RandomGraph.available_label)
        self.label = RandomGraph.available_label
        RandomGraph.available_label += 1

        # create nodes
        for node_id in range(num_nodes):
            G.add_node(node_id, label=node_label_generator())
        pprint.pprint(G.nodes(data=True))

        # create edges
        while G.number_of_edges() < num_edges:
            node_pair = numpy.random.choice(range(num_nodes),
                                            size=(2,),
                                            replace=False)
            G.add_edge(*node_pair, label=edge_label_generator())

        pprint.pprint(list(G.edges_iter(data=True)))
        self.graph = G

    def copy(self):
        pass

    def translate_by(self, offset):
        pass

    def add_random_node(self, connect_to):
        pass

    def __str__(self):
        graph_repr_lines = ['t # {}'.format(self.label)]

        # create a string of the form "v u_id label"
        #   where v is the literal character "v",
        #   u_id is the identifier for vertex u,
        #   and label is the label of vertex u
        for vid, v_data in self.graph.nodes(data=True):
            line = RandomGraph.vertex_line_format.format(
                vid=vid, label=v_data['label'])
            graph_repr_lines.append(line)

        # create a string of the form "e u_id v_id label",
        #   where e is the literal character "e",
        #   u_id is the identifier for vertex u,
        #   v_id is the identifier for vertex v,
        #   and label is the label associated with edge (u, v)
        for u, v, edge_data in self.graph.edges(data=True):
            line = RandomGraph.edge_line_format.format(
                uid=u, vid=v, label=edge_data['label'])
            graph_repr_lines.append(line)

        return '\n'.join(graph_repr_lines)

    def __repr__(self):
        return str(self)

    def _assert_legal_parameters(self, num_nodes, num_edges):
        if num_edges < num_nodes:
            raise ValueError('{0} edges is invalid for a graph with {1} '
                             'vertices'.format(num_edges, num_nodes))

        max_possible_edges = (num_nodes * (num_nodes-1))//2
        if num_edges > max_possible_edges:
            raise ValueError('A complete graph of {0} vertices has at most {1} '
                             'edges, so {2} is an invalid number of edges for '
                             'this graph.'.format(num_nodes, max_possible_edges,
                                             num_edges))


class GraphDatabaseWriter(object):
    def __init__(self, graphs):
        self.graphs = graphs

    def write_to(self, path):
        # enumerate each graph and associate an integer identifier with them
        reprs_for_all_graphs = []

        for g in self.graphs:
            reprs_for_all_graphs.append(str(g))

        reprs_for_all_graphs.append('t # {}'.format(-1))
        return '\n'.join(reprs_for_all_graphs)


class TestApproximateGraphIsomorphism(unittest.TestCase):

    def setUp(self):
        pass

    def test_approximate_subgraph_isomorphism(self):
        node_label_generator = lambda: numpy.random.randint(low=0,
                                                            high=100,
                                                            size=(2,))
        base_subgraph = RandomGraph(num_nodes=4,
                                    num_edges=5,
                                    node_label_generator=node_label_generator,
                                    edge_label_generator=int)
        print(base_subgraph)


if __name__ == '__main__':
    unittest.main()