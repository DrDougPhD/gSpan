import collections
import itertools
import copy
import time
import logging
import pprint

logger = logging.getLogger(__name__)

from graph import *


def record_timestamp(func):
    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        #self.timestamps[func.__name__ + '_c_in'] = time.clock()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()
        #self.timestamps[func.__name__ + '_c_out'] = time.clock()
    return deco


# TODO: get rid of this God class
class gSpan(object):
    def __init__(self, database_file_name,
                 min_support=10,
                 min_num_vertices=1,
                 max_num_vertices=float('inf'),
                 max_ngraphs=float('inf'),
                 is_undirected=True,
                 verbose=False,
                 visualize=False,
                 where=False):
        self.database_file_name = database_file_name
        self.graphs = dict()
        self.max_ngraphs = max_ngraphs
        self.is_undirected = is_undirected
        self.min_support = min_support
        self.min_num_vertices = min_num_vertices
        self.max_num_vertices = max_num_vertices
        self.DFScode = DFScode()
        self.support = 0
        self.frequent_size1_subgraphs = list()
        # include subgraphs with any num(but >= 2, <= max_num_vertices) of
        #  vertices
        self.frequent_subgraphs = list()
        self.counter = itertools.count()
        self.verbose = verbose
        self.visualize = visualize
        self.where = where
        self.timestamps = dict()
        if self.max_num_vertices < self.min_num_vertices:
            print('Max number of vertices can not be smaller than min number '
                  'of that.')
            print('Set max_num_vertices = min_num_vertices.')
            self.max_num_vertices = self.min_num_vertices

    @record_timestamp
    def run(self):
        self.read_graphs()
        self.generate_1edge_frequent_subgraphs()
        if self.max_num_vertices < 2:
            return

        root = collections.defaultdict(Projected)
        for gid, g in self.graphs.items():
            for vid, v in g.vertices.items():
                edges = self.get_forward_root_edges(g, vid)
                for e in edges:
                    root[(v.label, e.label, g.vertices[e.to].label)].append(
                        PDFS(gid, e, None))

        print('Length of root: {}'.format(len(root)))
        pprint.pprint(dict(root))

        #if self.verbose: print 'run:', root.keys()
        for vevlb, projected in root.items():
            self.DFScode.append(DFSedge(frm=0, to=1, vevlb=vevlb))
            self.subgraph_mining(projected)
            self.DFScode.pop()

    @record_timestamp
    def read_graphs(self):
        # TODO: each graph could have its own file
        # TODO: accept other graph file formats

        self.graphs = dict()
        with open(self.database_file_name) as f:
            lines = map(lambda x: x.strip(),
                        f.readlines())
            nlines = len(lines)

            tgraph, graph_cnt, edge_cnt = None, 0, 0
            for i, line in enumerate(lines):
                cols = line.split(' ')

                if cols[0] == 't':
                    if tgraph is not None:
                        self.graphs[graph_cnt] = tgraph
                        graph_cnt += 1
                        tgraph = None

                    if cols[-1] == '-1' or graph_cnt >= self.max_ngraphs:
                        # either the end of the file has been reached,
                        # or sufficiently many graphs have been read from the
                        #  file
                        break

                    tgraph = Graph(gid=graph_cnt,
                                   is_undirected=self.is_undirected,
                                   eid_auto_increment=True)

                elif cols[0] == 'v':
                    tgraph.add_vertex(cols[1], cols[2])

                elif cols[0] == 'e':
                    tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3])
                    
            if tgraph is not None:
                # adapt to input files that do not end with 't   -1'
                self.graphs[graph_cnt] = tgraph

        return self

    @record_timestamp
    def generate_1edge_frequent_subgraphs(self):
        vertex_label_counter = collections.Counter()
        vevlb_counter = collections.Counter()
        counted_vertex_labels = set()
        vevlb_counted = set()

        # Count the support of vertex labels and vertex-edge-vertex label
        # patterns across all databases.
        for g in self.graphs.values():
            for v in g.vertices.values():

                # If the vertex label has not yet been observed in this graph,
                # then increment the support of the vertex (label)
                # TODO: inefficient, better to use a hashmap
                if (g.gid, v.label) not in counted_vertex_labels:
                    vertex_label_counter[v.label] += 1

                # Make note that this particular vertex label was found in
                # this graph
                counted_vertex_labels.add((g.gid, v.label))

                # For each neighbor of the current vertex, if the
                # vertex-edge-vertex label triple has not already been
                # encountered in this graph, increment the support for it.
                for to, e in v.edges.items():
                    vlb1, vlb2 = v.label, g.vertices[to].label

                    if self.is_undirected and vlb1 > vlb2:
                        vlb1, vlb2 = vlb2, vlb1

                    if (g.gid, (vlb1, e.elb, vlb2)) not in vevlb_counter:
                        vevlb_counter[(vlb1, e.elb, vlb2)] += 1

                    # Make a note that this particular label triple has been
                    # encountered.
                    vevlb_counted.add((g.gid, (vlb1, e.label, vlb2)))

        # remove infrequent vertices or add frequent vertices
        for vertex_label, cnt in vertex_label_counter.items():
            if cnt >= self.min_support:
                g = Graph(gid=self.counter.next(),
                          is_undirected=self.is_undirected)
                g.add_vertex(0, vertex_label)
                self.frequent_size1_subgraphs.append(g)

                if self.min_num_vertices <= 1:
                    self.report_size1(g, support=cnt)

            else:
                continue
                for g in self.graphs.values():
                    # for each graph, remove vertices with the infrequent label
                    g.remove_vertex_with_vlb(vertex_label)

        if self.min_num_vertices > 1:
            self.counter = itertools.count()

        # remove edges of infrequent vev or ...
        for vevlb, cnt in vevlb_counter.items():
            if cnt >= self.min_support:
                continue
                # g = Graph(gid = self.counter.next(), is_undirected = self.is_undirected)
                # g.add_vertex(0, vevlb[0])
                # g.add_vertex(1, vevlb[2])
                # g.add_edge(0, 0, 1, vevlb[1])
                # self.frequent_subgraphs.append(g)
            else:
                continue
                for g in self.graphs.values():
                    g.remove_edge_with_vevlb(vevlb)
        #return copy.copy(self.frequent_subgraphs)

    def get_forward_root_edges(self, g, frm):
        # TODO: why are we obtaining only the forward edges?
        result = []
        v_frm = g.vertices[frm]

        print('.'*80)
        print('Vertex: {}'.format(v_frm))

        for to, e in v_frm.edges.items():

            print('Edge: {0} -- {1}'.format(e, type(e)))
            print('Points to Vertex: {0} -- {1}'.format(to, type(to)))

            if (not self.is_undirected) or v_frm.label <= g.vertices[to].label:
                result.append(e)

        return result

    def subgraph_mining(self, projected):
        self.support = self.get_support(projected)
        if self.support < self.min_support:
            #if self.verbose: print 'subgraph_mining: < min_support', self.DFScode
            return

        if not self.is_min():
            #if self.verbose: print 'subgraph_mining: not min'
            return

        self.report(projected)

        # construct the right-most path of the DFS tree
        right_most_path_bottom_up = self.DFScode.build_right_most_path()
        right_most_path = right_most_path_bottom_up[::-1]
        right_most_edge_index = right_most_path_bottom_up[0]
        right_most_vertex = self.DFScode[right_most_edge_index].to
        min_vertex_label = self.DFScode[0].vevlb[0]

        forward_root = collections.defaultdict(Projected)
        backward_root = collections.defaultdict(Projected)
        num_vertices = self.DFScode.get_num_vertices()
        for p in projected:
            g = self.graphs[p.gid]
            history = History(g, p)
            # backward
            for edge_index in right_most_path:
                e = self.get_backward_edge(g,
                                           history.edges[edge_index],
                                           history.edges[right_most_edge_index],
                                           history)
                print(type(e))
                if e is not None:
                    backward_root[
                        (self.DFScode[edge_index].frm, e.label)
                    ].append(PDFS(g.gid, e, p))

            # pure forward
            if num_vertices >= self.max_num_vertices:
                continue

            edges = self.get_forward_pure_edges(
                g, history.edges[right_most_edge_index],
                min_vertex_label,
                history)

            for e in edges:
                forward_root[
                    (right_most_vertex, e.label, g.vertices[e.to].label)
                ].append(PDFS(g.gid, e, p))

            # right_most_path forward
            for edge_index in right_most_path_bottom_up:
                edges = self.get_forward_rmpath_edges(g,
                                                      history.edges[edge_index],
                                                      min_vertex_label,
                                                      history)

                for e in edges:
                    forward_root[
                        (self.DFScode[edge_index].frm,
                         e.label,
                         g.vertices[e.to].label)
                    ].append(PDFS(g.gid, e, p))

        # backward
        for to, edge_label in backward_root:
            self.DFScode.append(
                DFSedge(right_most_vertex, to,
                        (VACANT_VERTEX_LABEL, edge_label, VACANT_VERTEX_LABEL)
            ))
            self.subgraph_mining(projected=backward_root[(to, edge_label)])
            self.DFScode.pop()

        # forward
        # if num_vertices >= self.max_num_vertices: # no need. because forward_root has no element.
        #     return
        for frm, edge_label, vertex_label in forward_root:
            self.DFScode.append(
                DFSedge(frm,
                        right_most_vertex+1,
                        (VACANT_VERTEX_LABEL, edge_label, vertex_label)
            ))
            self.subgraph_mining(forward_root[
                (frm, edge_label, vertex_label)
            ])
            self.DFScode.pop()

        return self

    def get_support(self, projected):
        return len(set([pdfs.gid for pdfs in projected]))

    def report_size1(self, g, support):
        g.display()
        print('\nSupport: {}'.format(support))
        print('-'*20)

    def report(self, projected):
        self.frequent_subgraphs.append(copy.copy(self.DFScode))
        if self.DFScode.get_num_vertices() < self.min_num_vertices:
            return
        g = self.DFScode.to_graph(gid = self.counter.next(), is_undirected = self.is_undirected)
        g.display()
        print('\nSupport: {}'.format(self.support))

        if self.visualize:
            g.plot()

        if self.where:
            print('where: {}'.format(list(set([p.gid for p in projected]))))

        print('-' * 20)

    def get_backward_edge(self, g, e1, e2, history):
        if self.is_undirected and e1 == e2:
            return None

        # gsize = g.get_num_vertices()
        # assert e1.frm >= 0 and e1.frm < gsize
        # assert e1.to >= 0 and e1.to < gsize
        # assert e2.to >= 0 and e2.to < gsize

        for to, e in g.vertices[e2.to].edges.items():
            if history.has_edge(e.eid) or e.to != e1.frm:
                continue

            # return e # ok? if reture here, then self.DFScodep[0] != DFScode_min[0] should be checked in is_min(). or:
            if self.is_undirected:
                if e1.label < e.label\
                    or (e1.label == e.label
                        and g.vertices[e1.to].label <= g.vertices[e2.to].label):
                    return e
            else:
                if g.vertices[e1.frm].label < g.vertices[e2.to]\
                    or (g.vertices[e1.frm].label == g.vertices[e2.to]
                        and e1.label <= e.label):
                    return e

            # if e1.elb < e.elb or (e1.elb == e.elb and g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
            #     return e

        return None

    def get_forward_pure_edges(self, g, rm_edge, min_vlb, history):
        result = []
        gsize = g.get_num_vertices()
        # assert rm_edge.to >= 0 and rm_edge.to < gsize
        for to, e in g.vertices[rm_edge.to].edges.items():
            # assert e.to >= 0 and e.to < gsize
            if min_vlb <= g.vertices[e.to].vlb and (not history.has_vertex(e.to)):
                result.append(e)
        return result

    def get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history):
        result = []
        gsize = g.get_num_vertices()
        # assert rm_edge.to >= 0 and rm_edge.to < gsize
        # assert rm_edge.frm >= 0 and rm_edge.frm < gsize
        to_vlb = g.vertices[rm_edge.to].vlb
        for to, e in g.vertices[rm_edge.frm].edges.items():
            new_to_vlb = g.vertices[to].vlb
            if rm_edge.to == e.to or min_vlb > new_to_vlb or history.has_vertex(e.to):
                continue
            # result.append(e) # ok? or:
            # if self.is_undirected:
            #     if rm_edge.elb < e.elb or (rm_edge.elb == e.elb and to_vlb <= new_to_vlb):
            #         return e
            # else:
            #     return e
            if rm_edge.elb < e.elb or (rm_edge.elb == e.elb and to_vlb <= new_to_vlb):
                result.append(e)
        return result

    def is_min(self):
        if self.verbose:
            print('is_min: checking {}'.format(self.DFScode))

        if len(self.DFScode) == 1:
            return True

        g = self.DFScode.to_graph(gid=VACANT_GRAPH_ID,
                                  is_undirected=self.is_undirected)
        DFScode_min = DFScode()
        root = collections.defaultdict(Projected)
        for vid, v in g.vertices.items():
            edges = self.get_forward_root_edges(g, vid)
            for e in edges:
                root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                    PDFS(g.gid, e, None))

        min_vevlb = min(root.keys())

        if self.verbose:
            print('is_min: bef p_is_min {0} {1} {2}'.format(
                min_vevlb,
                self.DFScode.get_num_vertices(),
                len(self.DFScode)))

        DFScode_min.append(DFSedge(0, 1, min_vevlb))
        # if self.DFScode[0] != DFScode_min[0]:
        #    # no need to check because of pruning in get_*_edge*
        #     return False

        def project_is_min(projected):
            right_most_path = DFScode_min.build_right_most_path()
            min_vlb = DFScode_min[0].vevlb[0]
            maxtoc = DFScode_min[right_most_path[0]].to

            backward_root = collections.defaultdict(Projected)
            flag, newto = False, 0,
            for i in range(len(right_most_path)-1,
                           0 if self.is_undirected else -1,
                           -1):
                if flag:
                    break

                for p in projected:
                    history = History(g, p)
                    e = self.get_backward_edge(
                        g,
                        history.edges[right_most_path[i]],
                        history.edges[right_most_path[0]],
                        history)

                    if e is not None:
                        # if self.verbose:
                        #     print('project_is_min: 6 {0} {1}'.format(e.frm,
                        #                                              e.to))

                        backward_root[e.elb].append(PDFS(g.gid, e, p))
                        newto = DFScode_min[right_most_path[i]].frm
                        flag = True

            # if self.verbose:
            #     print('project_is_min: 1 {0} {1} {2}'.format(
            #         flag,
            #         DFScode_min.get_num_vertices(),
            #         len(DFScode_min)))

            if flag:
                backward_min_elb = min(backward_root.keys())
                DFScode_min.append(
                    DFSedge(maxtoc,
                            newto,
                            (VACANT_VERTEX_LABEL,
                             backward_min_elb,
                             VACANT_VERTEX_LABEL)))

                idx = len(DFScode_min) - 1
                #if self.verbose: print 'project_is_min: 5', idx, len(self.DFScode)
                if self.DFScode[idx] != DFScode_min[idx]:
                    return False

                return project_is_min(backward_root[backward_min_elb])

            forward_root = collections.defaultdict(Projected)
            flag, newfrm = False, 0
            for p in projected:
                history = History(g, p)
                edges = self.get_forward_pure_edges(
                    g,
                    history.edges[right_most_path[0]],
                    min_vlb, history)

                if len(edges) > 0:
                    flag = True
                    newfrm = maxtoc
                    for e in edges:
                        forward_root[(e.elb, g.vertices[e.to].vlb)].append(
                            PDFS(g.gid, e, p))

            #if self.verbose: print 'project_is_min: 2', flag
            for rmpath_i in right_most_path:
                if flag:
                    break

                for p in projected:
                    history = History(g, p)
                    edges = self.get_forward_rmpath_edges(
                        g,
                        history.edges[rmpath_i],
                        min_vlb,
                        history)

                    if len(edges) > 0:
                        flag = True
                        newfrm = DFScode_min[rmpath_i].frm
                        for e in edges:
                            forward_root[(e.elb, g.vertices[e.to].vlb)].append(
                                PDFS(g.gid, e, p))

            #if self.verbose: print 'project_is_min: 3', flag

            if not flag:
                return True

            forward_min_evlb = min(forward_root.keys())
            #if self.verbose: print 'project_is_min: 4', forward_min_evlb, newfrm
            DFScode_min.append(DFSedge(
                newfrm,
                maxtoc + 1,
                (VACANT_VERTEX_LABEL, forward_min_evlb[0], forward_min_evlb[1])
            ))
            idx = len(DFScode_min) - 1
            if self.DFScode[idx] != DFScode_min[idx]:
                return False
            return project_is_min(forward_root[forward_min_evlb])

        res = project_is_min(root[min_vevlb])
        #if self.verbose: print 'is_min: leave'
        return res

    def time_stats(self):
        func_names = ['read_graphs', 'run']
        time_deltas = collections.defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = round(self.timestamps[fn + '_out']
                                    - self.timestamps[fn + '_in'], 2)
            #time_deltas[fn + '_c'] = round(self.timestamps[fn + '_c_out']
            #                       - self.timestamps[fn + '_c_in'], 2)
        print('Read:\t{} s'.format(
            time_deltas['read_graphs']))
        # , time_deltas['read_graphs_c'])

        print('Mine:\t{} s'.format(
            time_deltas['run'] - time_deltas['read_graphs']))
        # , time_deltas['run_c'] - time_deltas['read_graphs_c'])
        print('Total:\t{} s'.format(
            time_deltas['run']))
        # , time_deltas['run_c'])
        return self



class DFSedge(object):
    def __init__(self, frm, to, vevlb):
        self.frm = frm
        self.to = to
        self.vevlb = vevlb

    def __eq__(self, other):
        return self.frm == other.frm\
               and self.to == other.to\
               and self.vevlb == other.vevlb

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '({0: >3}, {1: >3}, {2: >3}, {3: >3}, {4: >3})'.format(
            self.frm, self.to, *self.vevlb)


class DFScode(list):
    """
    DFScode is a list of DFSedge.
    """
    def __init__(self):
        self.right_most_path = list()

    def __eq__(self, other):
        la, lb = len(self), len(other)
        if la != lb:
            return False
        for i in range(la):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return ''.join(['[', ','.join([str(dfsedge) for dfsedge in self]), ']'])

    def push_back(self, frm, to, vevlb):
        self.append(DFSedge(frm, to, vevlb))
        return self

    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)

        for dfsedge in self:
            frm, to = dfsedge.frm, dfsedge.to
            from_vertex_label, edge_label, to_vertex_label = dfsedge.vevlb

            if from_vertex_label != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, from_vertex_label)

            if to_vertex_label != VACANT_VERTEX_LABEL:
                g.add_vertex(to, to_vertex_label)

            g.add_edge(AUTO_EDGE_ID, frm, to, edge_label)

        return g

    def from_graph(self, g):
        pass

    def build_right_most_path(self):
        """
        Starting from the right-most vertex, construct the right-most path
        in a bottom-up manner by iterating through each element of the DFS
        code.
        :return: 
        """
        self.right_most_path = list()
        old_from_vertex = None

        for i in range(len(self)-1, -1, -1):
            dfsedge = self[i]
            frm, to = dfsedge.frm, dfsedge.to
            if frm < to and (old_from_vertex is None or to == old_from_vertex):
                self.right_most_path.append(i)
                old_from_vertex = frm
        return self.right_most_path

    def get_num_vertices(self):
        from_vertices = [dfsedge.frm for dfsedge in self]
        to_vertices = [dfsedge.to for dfsedge in self]
        return len(set(from_vertices + to_vertices))


class PDFS(object):
    def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
        self.gid = gid
        self.edge = edge
        self.prev = prev

    def __str__(self):
        return '<PartialDFS(graph: {0.gid: >4}, ' \
                           'edge: {0.edge}, ' \
                           'predecessor: {1})>'.format(
            self, bool(self.prev)
        )

    def __repr__(self):
        return str(self)


class Projected(list):
    """docstring for Projected
    Projected is a list of PDFS. Each element of Projected is a mapping 
    of one frequent subgraph onto one original graph.
    """
    def __init__(self):
        super(Projected, self).__init__()

    def push_back(self, gid, edge, prev):
        self.append(PDFS(gid, edge, prev))
        return self


class History(object):
    """docstring for History"""
    def __init__(self, g, pdfs):
        super(History, self).__init__()
        self.edges = list()
        self.vertices_used = collections.defaultdict(int)
        self.edges_used = collections.defaultdict(int)
        if pdfs is None:
            return

        while pdfs:
            e = pdfs.edge
            self.edges.append(e)
            self.vertices_used[e.frm], self.vertices_used[e.to], self.edges_used[e.eid] = 1, 1, 1
            pdfs = pdfs.prev
        self.edges = self.edges[::-1]

    def has_vertex(self, vid):
        return self.vertices_used[vid] == 1

    def has_edge(self, eid):
        return self.edges_used[eid] == 1
