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

from .util import ERROR, WARN, INFO, YAY
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
        return HierarchicalGraph(self.gcs, self.cost, self.expanded, self.iteration, self.start_vertex, self.target_vertex)

    def display_graph(self, graph_name="temp") -> None:
        graphviz = self.gcs.GetGraphvizString()
        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        data.write_png(graph_name + ".png")
        # data.write_svg(graph_name + ".svg")

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
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = False
        options.preprocessing = False  # TODO Do I need to deal with this?
        
        # start_vertex = None
        # target_vertex = None
        # for v in self.gcs.Vertices():
        #     if v.name() == "start":
        #         start_vertex = v
        #     if v.name() == "target":
        #         target_vertex = v
        #     if start_vertex is not None and target_vertex is not None:
        #         break
        # assert start_vertex is not None and target_vertex is not None

        INFO("Solving...", verbose=verbose)
        solution_to_graph = self.gcs.SolveShortestPath(self.start_vertex.id(), self.target_vertex.id(), options)
        if not solution_to_graph.is_success():
            WARN("Couldn't solve, inspect the graph")
            self.display_graph(str(self.iteration))
            return float('inf'), None

            # self.display_graph()
            # raise Exception("Couldn't solve, inspect the graph")

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
        # self.graph_built = False
        # self.solution = None

        self.set_gen = SetTesselation(options)

        # name to vertex dictionary, populated as we populate the graph with vertices
        self.iteration = 0
        self.num_solves = 0

    def solve(self, start_state: Point, target_state: Point) -> None:
        self.start_state = start_state
        self.target_state = target_state
        # initialize the graph
        graph = self.get_initial_graph()
        rem = []
        # while not a path -- keep expanding
        while graph.not_fully_expanded or graph.is_not_path:
            # graph is a path -- let's expand a relation in it!
            if graph.is_path:
                # expand and implace
                next_relation_index, next_expansion = graph.pick_next_relation_to_expand()
                graph = self.expand_graph(graph, next_relation_index, next_expansion)

            self.num_solves += 1
            solution_cost, solution_vertices = graph.solve()
            INFO("Solving at " + str(self.num_solves) + " cost is " + str(solution_cost))
            
            if solution_cost != float('inf'):
                solution_graph = self.make_graph_from_vertices(graph, solution_cost, solution_vertices)
                # solution_graph.display_graph("s1")
                assert solution_graph.is_path, "Solution graph is not path"
                # generate a remainder graph from solution
                if graph.is_not_path:
                    rem_graph = self.subtract(graph, solution_graph)
                else:
                    # if orinal graph was a path -- subtracting solution split the graph. 
                    # don't do anything in this scenario.
                    rem_graph = None
            
                # find best rem cost
                best_rem_index = None
                best_rem_cost = float('inf')
                for rem_index in range(len(rem)):
                    if best_rem_index is None:
                        best_rem_index = rem_index
                        best_rem_cost = rem[rem_index].cost
                    else:
                        if rem[rem_index].cost < best_rem_cost:
                            best_rem_index = rem_index
                            best_rem_cost = rem[rem_index].cost
                
                # add rem_graph to rems
                if rem_graph is not None:
                    if not rem_graph.bad_graph:
                        rem += [rem_graph]

            # if rem_graph.iteration == 41:
            #     graph.display_graph("g")
            #     rem_graph.display_graph("r")
            #     solution_graph.display_graph("s")
            #     return
                
            # current solution cost is best
            if solution_cost < best_rem_cost * 1.0:
                graph = solution_graph
                print("best is current")
            else:
                assert best_rem_index is not None
                # need to swap for a different rem
                graph = rem[best_rem_index]
                rem = rem[:best_rem_index] + rem[best_rem_index+1:]
                if solution_cost != float('inf'):
                    rem += [solution_graph]
                print("falling back to " + str(graph.iteration))

            best_rem_cost = float('inf')
            for rem_index in range(len(rem)):
                if rem[rem_index].cost < best_rem_cost:
                    best_rem_cost = rem[rem_index].cost

        graph.display_graph("final_solution")
        YAY("Optimal cost is " + str(graph.cost))



            


    def subtract(self, graph: HierarchicalGraph, solution_graph: HierarchicalGraph):
        # TODO: problem: i am subtracting from previous graph
        self.iteration += 1
        rem_edges = []
        rem_vertices = []
        path = solution_graph.get_path()
        rem_graph = graph.copy()
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

        # for node_name in path:
        #     if node_name not in ("start", "target"):
        #         edges_in_and_out = []
        #         vertex_in_and_out = None
        #         for e in rem_graph.gcs.Edges():
        #             if e.u().name() == node_name:
        #                 vertex_in_and_out = e.u()
        #                 edges_in_and_out += [e]
        #             elif e.v().name() == node_name:
        #                 edges_in_and_out += [e]
        #         num_edges_in_and_out = len(edges_in_and_out)
        #         if num_edges_in_and_out == 2:
        #             for e in edges_in_and_out:
        #                 rem_edges.add(e)
        #             rem_vertices.add(vertex_in_and_out)
        #         elif num_edges_in_and_out < 2:
        #             raise Exception("very few edges???")

        for e_name in rem_edges:
            for e in rem_graph.gcs.Edges():
                if e.name() == e_name:
                    rem_graph.gcs.RemoveEdge(e)
                    break
        for v_name in rem_vertices:
            for v in rem_graph.gcs.Vertices():
                if v.name() == v_name:
                    rem_graph.gcs.RemoveVertex(v)
                    break
            
        rem_graph.cost = solution_graph.cost + 0.0001
        rem_graph.iteration = self.iteration
        return rem_graph


    def make_graph_from_vertices(self, graph, solution_cost, solution_vertices):
        solution_expanded = graph.expanded
        solution_iteration = graph.iteration

        # gcs: GraphOfConvexSets,
        # start_vertex,
        # target_vertex

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

    def expand_graph(self, old_graph: HierarchicalGraph, next_relation_index: int, next_expansion: str):
        assert old_graph.is_path, "expanding node in a old_graph that is not a path"
        self.iteration += 1
        start_rels = self.set_gen.construct_rels_representation_from_point(self.start_state.x(), next_expansion)
        target_rels = self.set_gen.construct_rels_representation_from_point(self.target_state.x(), next_expansion)
        start_relation = start_rels[next_relation_index]
        target_relation = target_rels[next_relation_index]

        graph_path = old_graph.get_path()
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

        return HierarchicalGraph(graph, old_graph.cost, next_expansion, self.iteration, start_vertex, target_vertex)

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
