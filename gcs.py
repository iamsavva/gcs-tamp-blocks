import typing as T

import numpy as np
import numpy.typing as npt

import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import Point, GraphOfConvexSets, HPolyhedron
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    Cost,
    L2NormCost,
    MathematicalProgramResult,
    LinearConstraint,
)
from tqdm import tqdm

from IPython.display import Image, display


class GCSforBlocks:
    block_dim: int  # number of dimensions that describe the block world
    num_blocks: int  # number of blocks
    horizon: int  # number of mode swithes
    lb: npt.NDArray  # lower bound on box that bounds the operational space
    ub: npt.NDArray  # upper bound on box bounds the operational space
    width: float = 1.0  # block width
    delta: float = width / 2  # half block width

    ###################################################################################
    # Properties and inits

    @property
    def num_modes(self) -> int:
        """
        Number of modes. For the case with no pushing, we have 1 mode for free motion and a mode per block for when grasping that block.
        The case with pushing will have many more modes.
        """
        return self.num_blocks + 1

    @property
    def num_gcs_sets(self) -> int:
        """
        Returns number of GCS sets; right now it's just the number of modes, but in general each mode will have multiple convex sets as part of it.
        """
        return self.num_modes * 1

    @property
    def state_dim(self) -> int:
        """
        Dimension of the state x optimized at each vertex.
        (number of blocks + gripper) x (dimension of the world)
        """
        return (self.num_blocks + 1) * self.block_dim

    def __init__(self, block_dim=1, num_blocks=2, horizon=5):
        self.block_dim = block_dim
        self.num_blocks = num_blocks
        self.horizon = horizon

        self.lb = np.zeros(self.state_dim)
        self.ub = 10.0 * np.ones(self.state_dim)

        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        self.graph_edges = self.define_edges_between_sets()

        self.name_to_vertex = dict()

    # add function to change start / end vertex (? in general connectivity can be tricky(=)

    # solve function

    # display graph function

    # display solution function: in text

    # disaply solution function: visually

    def solve(self, convex_relaxation = True):
        start_vertex = self.name_to_vertex["start"]
        target_vertex = self.name_to_vertex["target"]
        self.solution = self.gcs.SolveShortestPath(start_vertex, target_vertex)
        self.show_graph_diagram(self.solution)

    def show_graph_diagram(
        self,
        # filename: str = "temp",
        result: T.Optional[MathematicalProgramResult] = None,
    ) -> None:

        if result is not None:
            graphviz = self.gcs.GetGraphvizString(result, True, precision=1)
        else:
            graphviz = self.gcs.GetGraphvizString()
        data = pydot.graph_from_dot_data(graphviz)[0]
        plt = Image(data.create_png())
        display(plt)
        # data.write_svg(filename)

    ###################################################################################
    # Building the finite horizon GCS

    def build_the_graph(
        self,
        initial_state: Point,
        initial_set_id: int,
        final_state: Point,
        final_set_id: int,
        horizon: int = 5,
    ):
        """
        Build the GCS graph of horizon H from start to target nodes.
        """
        self.horizon = horizon
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False
        # populate the graph nodes layer by layer
        for layer in tqdm(range(self.horizon), desc="Adding layers: "):
            self.add_nodes_for_layer(layer)
        # add the start node
        self.add_start_node(initial_state, initial_set_id)
        # add the target node
        self.add_target_node(final_state, final_set_id)
        self.graph_built = True

    def add_nodes_for_layer(self, layer: int):
        """
        The GCS graph is a trellis diagram (finite horizon GCS); each layer contains an entire GCS graph of modes.
        Here we add the nodes, edges, constraints, and costs on the new layer.
        """
        for set_id in range(self.num_gcs_sets):
            # add new vertex
            new_vertex = self.add_vertex(
                self.get_convex_set_for_set_id(set_id),
                self.get_vertex_name(layer, set_id),
            )
            # edges and costs for layer 0 are populated in add_start_node
            if layer > 0:
                edges_in = self.get_edges_into_set(set_id)
                for left_vertex_set_id in edges_in:
                    # add an edge
                    left_vertex_name = self.get_vertex_name(
                        layer - 1, left_vertex_set_id
                    )
                    left_vertex = self.name_to_vertex[left_vertex_name]
                    self.add_edge(left_vertex, new_vertex, left_vertex_set_id)

    def add_start_node(self, start_state: Point, start_set_id: int) -> None:
        """
        Adds start node to the graph.
        """
        # check that point belongs to the corresponding mode
        convex_set = self.get_convex_set_for_set_id(start_set_id)
        assert convex_set.PointInSet(start_state.x())
        # add node to the graph
        start_vertex = self.add_vertex(start_state, "start")
        # get edges from start
        edges_out = self.get_edges_out_of_set(start_set_id)
        for right_vertex_set_id in edges_out:
            # add an edge
            right_vertex_name = self.get_vertex_name(0, right_vertex_set_id)
            right_vertex = self.name_to_vertex[right_vertex_name]
            self.add_edge(start_vertex, right_vertex, start_set_id)

    def add_target_node(self, target_state: Point, target_set_id: int) -> None:
        """
        Adds target node to the graph.
        """
        # check that point belongs to the corresponding mode
        mode_set = self.get_convex_set_for_set_id(target_set_id)
        assert mode_set.PointInSet(target_state.x())
        # add node to the graph
        target_vertex = self.add_vertex(target_state, "target")
        # get edges into target
        edges_in = self.get_edges_into_set(target_set_id)
        for left_vertex_set_id in edges_in:
            # add an edge
            left_vertex_name = self.get_vertex_name(self.horizon-1, left_vertex_set_id)
            left_vertex = self.name_to_vertex[left_vertex_name]
            self.add_edge(left_vertex, target_vertex, left_vertex_set_id)

    ###################################################################################
    # Populating edges, edge cost, and edge constraint

    def add_edge(
        self,
        left_vertex: GraphOfConvexSets.Vertex,
        right_vertex: GraphOfConvexSets.Vertex,
        left_vertex_set_id: int,
    ) -> None:
        """
        Add an edge between two vertices, as well as corresponding constraints and costs.
        """
        # add an edge
        edge_name = self.get_edge_name(left_vertex.name(), right_vertex.name())
        edge = self.gcs.AddEdge(left_vertex, right_vertex, edge_name)
        # add constraints
        self.add_constraints_on_edge(
            left_vertex_set_id, edge
        )
        # add cost on the edge
        self.add_gripper_movement_cost_on_edge(edge)

    def add_vertex(self, state: HPolyhedron, name: str) -> GraphOfConvexSets.Vertex:
        vertex = self.gcs.AddVertex(state, name)
        self.name_to_vertex[name] = vertex
        return vertex

    def add_gripper_movement_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        L2 norm cost on the movement of the gripper.
        """
        xu, xv = edge.xu(), edge.xv()
        #  gripper state is 0 to block_dim
        A = np.zeros((self.block_dim, 2 * self.state_dim))
        A[:, 0 : self.block_dim] = np.eye(self.block_dim)
        A[:, self.state_dim : self.state_dim + self.block_dim] = -np.eye(self.block_dim)
        b = np.zeros(self.block_dim)
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[Cost](cost, np.append(xv, xu)))

    def add_constraints_on_edge(
        self, left_vertex_set_id: int, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        Add orbital constraints on the edge and that the right vertex belongs to set of the left vertex
        """
        # add orbital constraints on the edge
        xu, xv = edge.xu(), edge.xv()
        left_mode = self.get_mode_from_set_id(left_vertex_set_id)
        A, b = self.get_constraint_for_orbit_of_mode(left_mode)
        constraints = eq(A.dot(np.append(xv, xu)), b)
        for c in constraints:
            edge.AddConstraint(c)
        # add constraint that right point is in the left set
        left_vertex_set = self.get_convex_set_for_set_id(left_vertex_set_id)
        print(left_vertex_set.b())
        print(-np.ones(2*self.state_dim)*1000)
        set_con = LinearConstraint(left_vertex_set.A(), -np.ones(2*self.state_dim)*1000, left_vertex_set.b())
        con = Binding[LinearConstraint](set_con, np.append(xv, xu))
        edge.AddConstraint( con, xv() )



    ###################################################################################
    # Generating convex sets per mode or per set in a mode

    def get_convex_set_for_set_id(self, set_id: int) -> HPolyhedron:
        """
        Returns convex set that corresponds to the given vertex.
        For the simple case of transparent blocks, this is equivivalent to the convex set that corresponds to the mode.
        """
        return self.get_convex_set_for_mode(set_id)

    def get_convex_set_for_mode(self, mode: int) -> HPolyhedron:
        """
        Returns polyhedron:
        lb <= x <= ub
        x_0 = x_k
        (last constrained dropped if k = 0); k is the mode number
        """
        k = mode
        assert k < self.num_modes
        A = np.vstack((-np.eye(self.state_dim), np.eye(self.state_dim)))
        b = np.hstack((-self.lb, self.ub))
        if k == 0:
            return HPolyhedron(A, b)
        else:
            eq_con = np.zeros((self.block_dim, self.state_dim))
            eq_con[:, 0 : self.block_dim] = np.eye(self.block_dim)
            eq_con[:, k * self.block_dim : (k + 1) * self.block_dim] = -np.eye(
                self.block_dim
            )
            A = np.vstack((A, eq_con, -eq_con))
            b = np.hstack((b, np.zeros(self.block_dim), np.zeros(self.block_dim)))
            return HPolyhedron(A, b)

    def get_constraint_for_orbit_of_mode(self, mode: int):
        """
        Orbital constraint.
        Delta x_i is zero for any i != 0, k; k is the mode number
        """
        k = mode
        assert k < self.num_modes
        # v - u
        A = np.hstack((np.eye(self.state_dim), -np.eye(self.state_dim)))
        bd = self.block_dim
        # v_0 - u_0 is not constrained
        A[0:bd, :] = np.zeros((bd, 2*self.state_dim))
        # v_k - u_k is not constrained
        A[k * bd : (k + 1) * bd, :] = np.zeros((bd, 2*self.state_dim))
        b = np.zeros(self.state_dim)
        return A, b

    ###################################################################################
    # Functions related to connectivity between modes and sets within modes

    def define_edges_between_sets(self):
        """
        Return a matrix that represents edges in a directed graph of modes.
        For this simple example, the matrix is hand-built.
        When IRIS is used, sets must be A -- clustered (TODO: do they?),
        B -- connectivity checked and defined automatically.
        """
        mat = np.zeros((self.num_gcs_sets, self.num_gcs_sets))
        # mode 0 is connected to any other mode
        mat[0, :] = np.ones(self.num_gcs_sets)
        # mode k is connected only to 0;
        # TODO: do i need k to k transitions? don't think so
        mat[:, 0] = np.ones(self.num_gcs_sets)
        return mat

    def get_edges_into_set(self, set_id: int):
        """
        Use the edge matrix to determine which edges go into the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets
        return [v for v in range(self.num_gcs_sets) if self.graph_edges[v, set_id] == 1]

    def get_edges_out_of_set(self, set_id: int):
        """
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets
        return [v for v in range(self.num_gcs_sets) if self.graph_edges[set_id, v] == 1]

    def get_mode_from_set_id(self, set_id: int) -> int:
        """
        Returns a mode to which the vertex belongs.
        In the simple case of transparent blocks, each vertex represents the full mode, so it's just the vertex itself.
        """
        assert 0 <= set_id < self.num_gcs_sets
        mode = set_id
        return mode

    ###################################################################################
    # Vertex and edge naming

    def get_vertex_name(self, layer: int, set_id: int) -> str:
        """ """
        return "M_" + str(layer) + "_" + str(set_id)

    def get_edge_name(self, left_vertex_name: str, right_vertex_name: str) -> str:
        return "E: " + left_vertex_name + " -> " + right_vertex_name
