# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

import pydot
from tqdm import tqdm
from IPython.display import Image, display
import time

# from PIL import Image as PIL_Image

import pydrake.geometry.optimization as opt  # pylint: disable=import-error
from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    GraphOfConvexSets,
    HPolyhedron,
    ConvexSet,
)
from pydrake.solvers import (  # pylint: disable=import-error, unused-import
    Binding,
    L2NormCost,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearCost,
)

from .util import ERROR, WARN, INFO, YAY, timeit
from .gcs_options import GCSforAutonomousBlocksOptions, EdgeOptAB
from .set_tesselation_2d import SetTesselation
from .gcs import GCSforBlocks


class HierarchicalGraph:
    @property
    def not_fully_expanded(self) -> bool:
        self.check_expansion_consistency()
        return "X" in self.expanded

    @property
    def bad_graph(self) -> bool:
        return len(self.gcs.Vertices()) == len(self.gcs.Edges()) + 1

    @property
    def is_path(self) -> bool:
        return len(self.gcs.Vertices()) == len(self.gcs.Edges()) + 1

    @property
    def is_not_path(self) -> bool:
        return not self.is_path

    def __init__(
        self,
        gcs: GraphOfConvexSets,
        cost: float,
        expanded: str,
        iteration: int,
        start_vertex,
        target_vertex
    ):
        self.gcs = gcs
        self.cost = cost
        self.expanded = expanded
        self.iteration = iteration
        self.start_vertex = start_vertex
        self.target_vertex = target_vertex
        if start_vertex.name() != "start":
            WARN(str(self.iteration))
            self.display_graph()
        assert start_vertex.name() == "start"
        assert target_vertex.name() == "target"

    def copy(self):
        # TODO: this doesn't do jack shit
        # better graph construction that doesn't fuck up a previous graph?
        # it's actually ok to fuck up the previous graph and remove vertices; implace it
        return HierarchicalGraph(self.gcs, self.cost, self.expanded, self.iteration, self.start_vertex, self.target_vertex)

    def display_graph(self, graph_name="temp") -> None:
        graphviz = self.gcs.GetGraphvizString()
        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_png(graph_name + ".png")

    def find_path_to_target( self,
        edges: T.List[GraphOfConvexSets.Edge],
        start_vertex) -> T.List[GraphOfConvexSets.Vertex]:
        """Given a set of active edges, find a path from start to target"""
        current_edge = [e for e in edges if e.u() == start_vertex][0]
        v = current_edge.v()
        target_reached = v.name() == "target"
        if target_reached:
            return [start_vertex] + [v]
        else:
            return [start_vertex] + self.find_path_to_target(edges, v)

    def get_path(self) -> T.List[str]:
        assert self.is_path, "Trying to get a path when the graph is not a path " + str(self.iteration) 
        # TODO: this is redundant
        # i want faster storage; don't want to iterate through these every time
        path = []
        v = [v for v in self.gcs.Vertices() if v.name() == "start"][0]
        path += [v.name()]
        edges = self.gcs.Edges()
        while v.name() != "target":
            edges_out = [e for e in edges if e.u() == v]
            assert len(edges_out) == 1, "Graph is not a path "
            v = edges_out[0].v()
            path += [v.name()]
        return path


    def pick_next_relation_to_expand(self) -> T.Tuple[int, str]:
        """
        Simple implementation: just expand relations one by one in order.
        TODO: there are probably much more effective orderings of relation expansions!
        TODO: investigate
        """
        assert self.not_fully_expanded, "Fully expanded and asking to expand a relation!"
        for index, relation in enumerate(self.expanded):
            if relation == "X": 
                return index, self.expanded[:index] + "Y" + self.expanded[index+1:]

    def check_relation_consistency(self, node_name:str) -> bool:
        """
        If relation is X in expanded -- it should be X in node; 
        If relation is not X in expanded -- it should be not X in node
        """
        for i, relation in enumerate(node_name):
            if self.expanded[i] == "X" and relation != "X":
                return False
            if self.expanded[i] != "X" and relation == "X":
                return False
        return True

    def check_expansion_consistency(self):
        """
        Each node in the graph should be expanded according to self.expanded
        """
        graph_names = [v.name() for v in self.gcs.Vertices()]
        for name in graph_names:
            if name not in ("start", "target"):
                assert self.check_relation_consistency(name), (
                    "Bad vertex in graph! rel: "
                    + name
                    + " expanded: "
                    + self.expanded
                    + " iteration: "
                    + str(self.iteration)
                    + "\n"
                    + " ".join(graph_names)
                )

    def get_solution_path(self, solution, start_vertex) -> T.Tuple[T.List[str], npt.NDArray]:
        """Given a solved GCS problem, and assuming it's tight, find a path from start to target"""
        # find edges with non-zero flow
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [solution.GetSolution(p) for p in flow_variables]
        active_edges = [
            edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow > 0.01
        ]
        return self.find_path_to_target(active_edges, start_vertex)


    def solve(self, verbose=False):
        # TODO: 
        # i only spend 3 seconds outside of this call
        # can i speed up this bit
        # can i get a second best path as well for a better estimate of cost to go
        # is preprocessing on my side? no, 76 vs 95 secs

        # don't solve problems that aren't feasible
        # problems aren't feasible if a set is empty
        # i know which sets are empty and which are not -- use chebyshev for this
        # this requires work for expand 

        # TODO: play with the convex relaxation
        # TODO: convex relaxation is almost 2x as fast!
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.preprocessing = False  # TODO Do I need to deal with this?
        # TODO: make sure this number is dependent on actual number of possible paths
        options.max_rounded_paths = 50

        INFO("Solving...", verbose=verbose)
        solution_to_graph = self.gcs.SolveShortestPath(self.start_vertex.id(), self.target_vertex.id(), options)
        if not solution_to_graph.is_success():
            WARN("Couldn't solve, inspect the graph")
            # self.display_graph(str(self.iteration))
            return float('inf'), None

        solution_vertices = self.get_solution_path(solution_to_graph, self.start_vertex)
        return solution_to_graph.get_optimal_cost(), solution_vertices

class HierarchicalGCSAB:
    """
    GCS for N-dimensional block moving using a top-down suction cup.
    """

    ###################################################################################
    # Building the finite horizon GCS

    def __init__(self, options: GCSforAutonomousBlocksOptions):
        # options
        self.opt = options

        # init the graph
        self.gcs = GraphOfConvexSets()
        self.set_gen = SetTesselation(options)

        # name to vertex dictionary, populated as we populate the graph with vertices
        self.iteration = 0
        self.num_solves = 0

        self.rem = []
        self.start_state = None
        self.target_state = None
        self.graph = None
        self.solve_dt = 0

    def add_to_rem(self, graph):
        if len(self.rem) == 0:
            self.rem = [graph]
            return
        i = 0
        j = len(self.rem)
        while i < j-1:
            t = int((i+j)/2)
            if self.rem[t].cost >= graph.cost:
                i = t
            else:
                j = t
        if self.rem[i].cost >= graph.cost:
            self.rem = self.rem[:i+1] + [graph] + self.rem[i+1:]
        else:
            self.rem = self.rem[:i] + [graph] + self.rem[i:]

    def solve(self, start_state: Point, target_state: Point) -> None:
        self.start_state = start_state
        self.target_state = target_state
        # initialize the graph
        self.graph = self.get_initial_graph()
        # while not a path -- keep expanding

        full_time = timeit()
        solve_time = timeit()

        while self.graph.not_fully_expanded or self.graph.is_not_path:
            # TODO: so you found A fully expanded path, what now? 
            # TODO: stop expanding only when all graphs in the rem have lower cost
            # you don't get an optimality certificate until you expand every other path

            # graph is a path -- let's expand a relation in it!
            if self.graph.is_path:
                # expand and implace
                next_relation_index, next_expansion = self.graph.pick_next_relation_to_expand()
                self.graph = self.expand_graph(next_relation_index, next_expansion)

            self.num_solves += 1
            solve_time.start()
            solution_cost, solution_vertices = self.graph.solve()
            solve_time.end()
            INFO("Solving at " + str(self.num_solves) + " cost is " + str(solution_cost))

            if solution_cost == float('inf'):
                # that problem was infeasible
                # backtrack
                self.graph = self.rem[-1]
                self.rem = self.rem[:-1]
                continue
                
            solution_graph = self.make_graph_from_vertices(self.graph, solution_cost, solution_vertices)
            # solution_graph.display_graph("s1")
            assert solution_graph.is_path, "Solution graph is not path"
            # generate a remainder graph from solution
            if self.graph.is_not_path:
                self.subtract(solution_graph)
            else:
                self.graph = None
            
            # add post-subtracted graph to rems
            if self.graph is not None and not self.graph.bad_graph:
                self.add_to_rem(self.graph)

            # current solution cost is best
            if len(self.rem) == 0 or solution_cost < self.rem[-1].cost * 1.0:
                self.graph = solution_graph
                print("best is current")
            else:
                # need to swap for a different rem
                self.graph = self.rem[-1]
                self.rem = self.rem[:-1]
                self.add_to_rem(solution_graph)
                print("falling back to " + str(self.graph.iteration))

        full_time.dt("Full run time")
        solve_time.total("solve time")
        self.graph.display_graph("final_solution")
        YAY("Optimal cost is " + str(self.graph.cost))


    def subtract(self, solution_graph: HierarchicalGraph):
        # TODO: more careful subtraction
        # maintain nodes 1-n in path? knowing which subpaths it does not contain?

        self.iteration += 1
        rem_edges = []
        rem_vertices = []
        path = solution_graph.get_path()
        # find the right node
        relation_index = solution_graph.expanded.find('X')-1
        prev_rel = None
        for i, node_name in enumerate(path):
            if node_name not in ("start", "target"):
                if prev_rel is None:
                    prev_rel = node_name[relation_index]
                else:
                    if prev_rel != node_name[relation_index]:
                        rem_edges += [path[i-1] + "_" + node_name]
                        prev_rel = node_name[relation_index]
        if len(rem_edges) > 2:
            raise Exception("removing too many edges mate")
        if len(rem_edges) == 2:
            rem_vertices += [rem_edges[1][:self.opt.rels_len]]
        if len(rem_edges) == 0:
            raise Exception("removing nothing")

        for e_name in rem_edges:
            for e in self.graph.gcs.Edges():
                if e.name() == e_name:
                    self.graph.gcs.RemoveEdge(e)
                    break
        for v_name in rem_vertices:
            for v in self.graph.gcs.Vertices():
                if v.name() == v_name:
                    self.graph.gcs.RemoveVertex(v)
                    break
            
        self.graph.cost = solution_graph.cost + 0.0001
        self.graph.iteration = self.iteration


    def make_graph_from_vertices(self, graph, solution_cost, solution_vertices):
        # do i need to
        # option 2: instead store which edges are active and which ones are not
        # in an array
        # though this may get somewhat messy...
        # can't remove edges, but can always add more

        solution_expanded = graph.expanded
        solution_iteration = graph.iteration

        solution_graph = GraphOfConvexSets()
        prev_node = None
        for node in solution_vertices:
            solution_node = solution_graph.AddVertex( node.set(), node.name() )
            if node.name() == "start":
                solution_start_vertex = solution_node
                prev_node = solution_node
            elif node.name() == "target":
                solution_target_vertex = solution_node
                self.add_edge(solution_graph, prev_node, solution_node, EdgeOptAB.target_edge())
            else:
                if prev_node.name() == "start":
                    self.add_edge(solution_graph, prev_node, solution_node, EdgeOptAB.equality_edge())
                else:
                    self.add_edge(solution_graph, prev_node, solution_node, EdgeOptAB.move_edge())
                prev_node = solution_node
        return HierarchicalGraph(solution_graph, solution_cost, solution_expanded, solution_iteration, solution_start_vertex, solution_target_vertex)

    def expand_graph(self, next_relation_index: int, next_expansion: str):
        # TODO: do not exapnd nodes that are useless 
        # more efficient way to store individual nodes
        # don't repeat subgraphs
        # i need smth like "set all flows to zero, set mine to non-zero"


        assert self.graph.is_path, "expanding node in a old_graph that is not a path"
        self.iteration += 1
        start_rels = self.set_gen.construct_rels_representation_from_point(self.start_state.x(), next_expansion)
        target_rels = self.set_gen.construct_rels_representation_from_point(self.target_state.x(), next_expansion)
        start_relation = start_rels[next_relation_index]
        target_relation = target_rels[next_relation_index]

        graph_path = self.graph.get_path()
        graph = GraphOfConvexSets()

        start_col_v = None
        target_col_v = None
        start_vertex = None
        target_vertex = None
        # expanding all new nodes
        for node in graph_path:
            if node == "start":
                start_vertex = self.add_vertex(graph, self.start_state, "start")
                start_col_v = start_vertex
            elif node == "target":
                assert target_col_v is not None, "target column is none mate this is wrong"
                target_vertex = self.add_vertex(graph, self.target_state, "target")
                self.add_edge(graph, target_col_v, target_vertex, EdgeOptAB.target_edge())
            else:
                grounded_start_name = node[:next_relation_index] + start_relation + node[next_relation_index+1:]
                grounded_start_vertex = self.add_vertex(graph, self.set_gen.get_set_for_rels(grounded_start_name), grounded_start_name)
                # connect with previous column
                assert start_col_v is not None
                if start_col_v.name() == "start":
                    self.add_edge(graph, start_col_v, grounded_start_vertex, EdgeOptAB.equality_edge())
                else:
                    self.add_edge(graph, start_col_v, grounded_start_vertex, EdgeOptAB.move_edge())
                start_col_v = grounded_start_vertex

                if start_relation == target_relation:
                    target_col_v = start_col_v
                    continue
                elif target_relation in self.opt.rel_nbhd[start_relation]:
                    grounded_target_name = node[:next_relation_index] + target_relation + node[next_relation_index+1:]
                    grounded_target_vertex = self.add_vertex(graph, self.set_gen.get_set_for_rels(grounded_target_name), grounded_target_name)
                    self.add_edge(graph, grounded_start_vertex, grounded_target_vertex, EdgeOptAB.move_edge())
                    if target_col_v is not None:
                        self.add_edge(graph, target_col_v, grounded_target_vertex, EdgeOptAB.move_edge())
                    target_col_v = grounded_target_vertex
                else:
                    nbh = self.opt.rel_nbhd[start_relation]
                    grounded_nbh_0_name = node[:next_relation_index] + nbh[0] + node[next_relation_index+1:]
                    grounded_nbh_1_name = node[:next_relation_index] + nbh[1] + node[next_relation_index+1:]
                    grounded_nbh_0_vertex = self.add_vertex(graph, self.set_gen.get_set_for_rels(grounded_nbh_0_name), grounded_nbh_0_name)
                    grounded_nbh_1_vertex = self.add_vertex(graph, self.set_gen.get_set_for_rels(grounded_nbh_1_name), grounded_nbh_1_name)
                    grounded_target_name = node[:next_relation_index] + target_relation + node[next_relation_index+1:]
                    grounded_target_vertex = self.add_vertex(graph, self.set_gen.get_set_for_rels(grounded_target_name), grounded_target_name)

                    self.add_edge(graph, grounded_start_vertex, grounded_nbh_0_vertex, EdgeOptAB.move_edge())
                    self.add_edge(graph, grounded_start_vertex, grounded_nbh_1_vertex, EdgeOptAB.move_edge())
                    self.add_edge(graph, grounded_nbh_0_vertex, grounded_target_vertex, EdgeOptAB.move_edge())
                    self.add_edge(graph, grounded_nbh_1_vertex, grounded_target_vertex, EdgeOptAB.move_edge())

                    if target_col_v is not None:
                        self.add_edge(graph, target_col_v, grounded_target_vertex, EdgeOptAB.move_edge())
                    target_col_v = grounded_target_vertex
        
        # for v in graph.Vertices():
        #     if v not in ("start", "target"):
        #         if v.name() not in self.set_gen.rels2set:
        #             graph.RemoveVertex(v)
                    # TODO: this is much more complicated, investigate

        return HierarchicalGraph(graph, self.graph.cost, next_expansion, self.iteration, start_vertex, target_vertex)

    def get_initial_graph(self):
        graph = GraphOfConvexSets()
        # add start and target
        start_vertex = self.add_vertex(graph, self.start_state, "start")
        target_vertex = self.add_vertex(graph, self.target_state, "target")
        # know nothing vertex -- only vertex in init graph; fully relaxed relations
        xxx_rels = "X" * self.opt.rels_len
        xxx_set = self.set_gen.get_set_for_rels(xxx_rels)
        xxx_vertex = self.add_vertex(graph, xxx_set, xxx_rels)
        # start -> know_nothing -> target
        self.add_edge(graph, start_vertex, xxx_vertex, EdgeOptAB.equality_edge())
        self.add_edge(graph, xxx_vertex, target_vertex, EdgeOptAB.target_edge())
        # return the useful hierarchical graph representation
        return HierarchicalGraph(graph, float("inf"), xxx_rels, self.iteration, start_vertex, target_vertex)

    ###################################################################################
    # Adding edges and vertices

    def add_vertex(
        self, graph: GraphOfConvexSets, convex_set: HPolyhedron, name: str
    ) -> None:
        """
        Define a vertex with a convex set.
        """
        # check myself
        vertex_names = [v.name() for v in graph.Vertices()]
        assert name not in vertex_names, (
            "Adding vertex again! Vertex: "
            + name
            + "\nAlready in: "
            + str(vertex_names)
        )
        vertex = graph.AddVertex(convex_set, name)
        return vertex

    def add_edge(
        self,
        graph: GraphOfConvexSets,
        left_vertex: GraphOfConvexSets.Vertex,
        right_vertex: GraphOfConvexSets.Vertex,
        edge_opt: EdgeOptAB,
    ) -> None:
        """
        READY
        Add an edge between two vertices, as well as corresponding constraints and costs.
        """
        # add an edge
        edge_name = self.get_edge_name(left_vertex.name(), right_vertex.name())
        edge = graph.AddEdge(left_vertex, right_vertex, edge_name)

        # -----------------------------------------------------------------
        # Adding constraints
        # -----------------------------------------------------------------
        if edge_opt.add_set_transition_constraint:
            self.add_common_set_at_transition_constraint(left_vertex.set(), edge)
        if edge_opt.add_equality_constraint:
            self.add_point_equality_constraint(edge)
        # -----------------------------------------------------------------
        # Adding costs
        # -----------------------------------------------------------------
        # add movement cost on the edge
        if edge_opt.add_each_block_movement_cost:
            # self.add_each_block_movement_cost(edge)
            self.add_full_movement_cost(edge)

    ###################################################################################
    # Adding constraints and cost terms

    def add_common_set_at_transition_constraint(
        self, left_vertex_set: HPolyhedron, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # fill in linear constraint on the right vertex
        A = left_vertex_set.A()
        lb = -np.ones(left_vertex_set.b().size) * 1000
        ub = left_vertex_set.b()
        set_con = LinearConstraint(A, lb, ub)
        edge.AddConstraint(Binding[LinearConstraint](set_con, edge.xv()))

    def add_point_equality_constraint(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # get set that corresponds to left vertex
        A = np.hstack((np.eye(self.opt.state_dim), -np.eye(self.opt.state_dim)))
        b = np.zeros(self.opt.state_dim)
        set_con = LinearEqualityConstraint(A, b)
        edge.AddConstraint(
            Binding[LinearEqualityConstraint](set_con, np.append(edge.xu(), edge.xv()))
        )

    def add_each_block_movement_cost(self, edge: GraphOfConvexSets.Edge) -> None:
        xu, xv = edge.xu(), edge.xv()
        for i in range(self.opt.num_blocks):
            d = self.opt.block_dim
            n = self.opt.state_dim
            A = np.zeros((d, 2 * n))
            A[:, i * d : i * d + d] = np.eye(d)
            A[:, n + i * d : n + i * d + d] = -np.eye(d)
            b = np.zeros(d)
            # add the cost
            cost = L2NormCost(A, b)
            edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    def add_full_movement_cost(self, edge):
        xu, xv = edge.xu(), edge.xv()
        n = self.opt.state_dim
        A = np.hstack((np.eye(n), -np.eye(n)))
        b = np.zeros(n)
        # add the cost
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    ###################################################################################

    def get_edge_name(self, left_vertex_name: str, right_vertex_name: str) -> str:
        return left_vertex_name + "_" + right_vertex_name
