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



class GCSAutonomousBlocks(GCSforBlocks):
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
        self.graph_built = False

        self.set_gen = SetTesselation(options)

        # name to vertex dictionary, populated as we populate the graph with vertices
        self.name_to_vertex = dict()  # T.Dict[str, GraphOfConvexSets.Vertex]


    def build_the_graph(
        self,
        start_state: Point,
        target_state: Point,
    ) -> None:
        """
        Build the GCS graph of horizon H from start to target nodes.
        TODO:
        - allow target state to be a set
        """
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        # add all vertices
        self.add_all_vertices(start_state, target_state)

        # add all edges
        self.add_all_edges(start_state, target_state)

        self.graph_built = True

    ###################################################################################
    # Adding layers of nodes (trellis diagram style)

    def add_all_vertices(
        self,
        start_state: Point,
        target_state: Point,
    ) -> None:
        self.add_vertex(start_state, "start")
        for dir in self.set_gen.dir2set:
            self.add_vertex(self.set_gen.dir2set[dir], dir)
        self.add_vertex(target_state, "target")

    def add_all_edges(self, start_state: Point, target_state: Point,) -> None:
        ############################
        start_set = self.set_gen.construct_dir_representation_from_point(start_state.x())
        self.connect_vertices("start", start_set, EdgeOptAB.equality_edge())

        target_set = self.set_gen.construct_dir_representation_from_point(target_state.x())
        self.connect_vertices(target_set, "target", EdgeOptAB.target_edge())

        for dir in self.set_gen.dir2set:
            nbhd = self.set_gen.get_1_step_neighbours(dir)
            for nbh in nbhd:
                self.connect_vertices(dir, nbh, EdgeOptAB.move_edge())

    ###################################################################################
    # Populating edges and vertices

    def add_edge(
        self,
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
        edge = self.gcs.AddEdge(left_vertex, right_vertex, edge_name)

        # -----------------------------------------------------------------
        # Adding constraints
        # -----------------------------------------------------------------
        if edge_opt.add_set_transition_constraint:
            left_set = self.set_gen.dir2set[left_vertex.name()]
            self.add_common_set_at_transition_constraint(left_set, edge)
        if edge_opt.add_equality_constraint:
            self.add_point_equality_constraint(edge)
        # -----------------------------------------------------------------
        # Adding costs
        # -----------------------------------------------------------------
        # add movement cost on the edge
        if edge_opt.add_each_block_movement_cost:
            # self.add_each_block_movement_cost(edge)
            self.add_full_movement_cost(edge)


    def add_vertex(
        self, convex_set: HPolyhedron, name: str
    ) -> GraphOfConvexSets.Vertex:
        """
        Define a vertex with a convex set.
        """
        # create a vertex
        vertex = self.gcs.AddVertex(convex_set, name)
        self.name_to_vertex[name] = vertex
        return vertex

    ###################################################################################
    # Adding constraints and cost terms

    def add_common_set_at_transition_constraint(
        self, left_vertex_set: HPolyhedron, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        READY
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # fill in linear constraint on the right vertex
        A = left_vertex_set.A()
        lb = -np.ones(left_vertex_set.b().size) * 1000
        ub = left_vertex_set.b()
        set_con = LinearConstraint(A, lb, ub)
        edge.AddConstraint(Binding[LinearConstraint](set_con, edge.xv()))

    def add_each_block_movement_cost(self, edge: GraphOfConvexSets.Edge) -> None:
        xu, xv = edge.xu(), edge.xv()
        for i in range(self.opt.num_blocks):
            d = self.opt.block_dim
            n = self.opt.state_dim
            A = np.zeros((d, 2 * n))
            A[:, i*d:i*d+d] = np.eye(d)
            A[:, n+i*d: n+i*d+d] = -np.eye(d)
            b = np.zeros(d)
            # add the cost
            cost = L2NormCost(A, b)
            edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    def add_full_movement_cost(self, edge):
        xu, xv = edge.xu(), edge.xv()
        n = self.opt.state_dim
        A = np.hstack( (np.eye(n), -np.eye(n)) )
        b = np.zeros(n)
        # add the cost
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))


    ###################################################################################

    def get_edge_name(self, left_vertex_name: str, right_vertex_name: str) -> str:
        return left_vertex_name + "_" + right_vertex_name

    ###################################################################################
    # Solve and display solution

    def get_solution_path(self) -> T.Tuple[T.List[str], npt.NDArray]:
        """Given a solved GCS problem, and assuming it's tight, find a path from start to target"""
        assert self.graph_built, "Must build graph first!"
        assert self.solution.is_success(), "Solution was not found"
        # find edges with non-zero flow
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [self.solution.GetSolution(p) for p in flow_variables]

        not_tight = np.any(
            np.logical_and(0.05 < np.array(flow_results), np.array(flow_results) < 0.95)
        )
        if not_tight:
            WARN("CONVEX RELAXATION NOT TIGHT")
            return

        active_edges = [
            edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow >= 0.99
        ]
        # using these edges, find the path from start to target
        path = self.find_path_to_target(active_edges, self.name_to_vertex["start"])
        sets = [v.name() for v in path]
        vertex_values = np.vstack([self.solution.GetSolution(v.x()) for v in path])
        return sets, vertex_values

    # def verbose_solution_description(self) -> None:
    #     """Describe the solution in text: grasp X, move to Y, ungrasp Z"""
    #     assert self.solution.is_success(), "Solution was not found"
    #     modes, vertices = self.get_solution_path()
    #     for i in range(len(vertices)):
    #         vertices[i] = ["%.2f" % v for v in vertices[i]]
    #     mode_now = modes[1]
    #     INFO("-----------------------")
    #     INFO("Solution is:")
    #     INFO("-----------------------")
    #     for i in range(len(modes)):  # pylint: disable=consider-using-enumerate
    #         sg = vertices[i][0 : self.opt.block_dim]
    #         if modes[i] == "start":
    #             INFO("Start at", sg)
    #         elif modes[i] == "target":
    #             INFO("Move to", sg, "; Finish")
    #         else:
    #             mode_next = modes[i]
    #             if mode_next == mode_now:
    #                 grasp = ""
    #             elif mode_next == "0":
    #                 grasp = "Ungrasp block " + str(mode_now)
    #             else:
    #                 grasp = "Grasp block " + str(mode_next)
    #             mode_now = mode_next
    #             INFO("Move to", sg, "; " + grasp)
