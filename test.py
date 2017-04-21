import pprint

import networkx
import numpy
numpy.random.seed(0)
import unittest

class RandomGraph(object):
    def __init__(self, num_nodes, num_edges, node_label_generator,
                 edge_label_generator):
        self._assert_legal_parameters(num_nodes, num_edges)

        G = networkx.Graph()
        for node_id in range(num_nodes):
            G.add_node(node_id, label=node_label_generator())
        pprint.pprint(G.nodes(data=True))

    def copy(self):
        pass

    def translate_by(self, offset):
        pass

    def add_random_node(self, connect_to):
        pass

    def __str__(self):
        return 'some graph'

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
        pass


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


if __name__ == '__main__':
    unittest.main()