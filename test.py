import pprint
import itertools
import networkx
import numpy
numpy.random.seed(0)
import unittest

class RandomGraph(object):
    vertex_line_format = 'v {vid} {label}'
    edge_line_format = 'e {uid} {vid} {label}'
    available_label = 0

    def __init__(self, G=None):
        self.label = RandomGraph.available_label
        RandomGraph.available_label += 1
        if G is not None:
            self.graph = G.copy()
            self.graph.graph['label'] = self.label
            self.next_node_id = G.number_of_nodes()

        else:
            self.graph = networkx.Graph(label=self.label)
            self.next_node_id = 0

        self.node_label_generator = None
        self.edge_label_generator = None

    def create(self, num_nodes, num_edges, node_label_generator,
                 edge_label_generator):
        self._assert_legal_parameters(num_nodes, num_edges)

        self.node_label_generator = node_label_generator
        self.edge_label_generator = edge_label_generator

        G = self.graph

        # create nodes
        for node_id in range(num_nodes):
            G.add_node(node_id, label=node_label_generator())

        self.next_node_id = G.number_of_nodes()

        # pprint.pprint(G.nodes(data=True))

        # create edges
        while G.number_of_edges() < num_edges:
            node_pair = numpy.random.choice(range(num_nodes),
                                            size=(2,),
                                            replace=False)
            G.add_edge(*node_pair, label=edge_label_generator())

        # pprint.pprint(list(G.edges_iter(data=True)))
        return self

    def copy(self):
        cloned = RandomGraph(G=self.graph)
        cloned.edge_label_generator = self.edge_label_generator
        cloned.node_label_generator = self.node_label_generator
        return cloned

    def translate_by(self, offset):
        pass

    def add_random_nodes(self, how_many=1, connect_to_num_nodes=1):
        G = self.graph

        # generate new nodes
        starting_node_index = self.next_node_id
        new_node_indices = list(range(starting_node_index, starting_node_index+how_many))
        for node_id in new_node_indices:
            G.add_node(node_id, label=self.node_label_generator())
            self.last_node_id = node_id

        self.next_node_id = G.number_of_nodes()

        # connect them up to pre-existing nodes
        final_edge_count = G.number_of_edges() + connect_to_num_nodes
        print(new_node_indices)
        possible_edges = list(itertools.product(G.nodes(),
                                                new_node_indices))
        chosen_edges = None
        print(possible_edges)
        while G.number_of_edges() < final_edge_count:
            to_node = numpy.random.choice(new_node_indices)
            from_node = numpy.random.choice(G.nodes())
            G.add_edge(from_node, to_node, label=self.edge_label_generator())

    def __str__(self):
        graph_repr_lines = ['t # {}'.format(self.label)]

        # create a string of the form "v u_id label"
        #   where v is the literal character "v",
        #   u_id is the identifier for vertex u,
        #   and label is the label of vertex u
        for vid, v_data in self.graph.nodes(data=True):
            node_label = '({})'.format(','.join(
                map(str, v_data['label'])))
            line = RandomGraph.vertex_line_format.format(
                vid=vid, label=node_label)
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
        self.base_subgraph = []
        self.exact_subgraph_isomorphisms = []

    def test_exact_subgraph_isomorphism(self):
        pass

    def test_approximate_subgraph_isomorphism(self):
        node_label_generator = lambda: numpy.random.randint(low=0,
                                                            high=100,
                                                            size=(2,))
        base_subgraph = RandomGraph().create(
            num_nodes=4,
            num_edges=5,
            node_label_generator=node_label_generator,
            edge_label_generator=int)

        graph1 = base_subgraph.copy()
        graph1.translate_by(offset=numpy.array([1,0]))
        graph1.add_random_nodes(how_many=5, connect_to_num_nodes=4)

        print(base_subgraph)
        print(graph1)


if __name__ == '__main__':
    unittest.main()