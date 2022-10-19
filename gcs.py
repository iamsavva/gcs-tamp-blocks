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
    # L2NormCostUsingConicConstraint,
    MathematicalProgramResult,
    LinearConstraint,
    LinearEqualityConstraint,
)
from tqdm import tqdm

from IPython.display import Image, display

# import colorama
from colorama import Fore


def ERROR(*texts):
    print(Fore.RED + " ".join([str(text) for text in texts]))


def WARN(*texts):
    print(Fore.YELLOW + " ".join([str(text) for text in texts]))


def INFO(*texts):
    print(Fore.BLUE + " ".join([str(text) for text in texts]))


def YAY(*texts):
    print(Fore.GREEN + " ".join([str(text) for text in texts]))


def make_simple_transparent_gcs_test(
    block_dim, num_blocks, horizon, max_rounded_paths=30
):
    gcs = GCSforBlocks(block_dim, num_blocks, horizon)

    width = 1
    ub = width * 2 * (num_blocks + 1)
    gcs.set_block_width(width)
    gcs.set_ub(ub)

    initial_state = []
    for i in range(gcs.num_modes):
        block_state = [0] * gcs.block_dim
        block_state[0] = width * (2 * i + 1)
        initial_state += block_state
    initial_point = Point(np.array(initial_state))
    final_state = []
    for i in range(gcs.num_modes):
        block_state = [0] * gcs.block_dim
        block_state[-1] = ub - width * (2 * i + 1)
        final_state += block_state
    final_point = Point(np.array(final_state))
    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve(max_rounded_paths=max_rounded_paths)
    gcs.verbose_solution_description()
    gcs.display_graph()


class GCSforBlocks:
    block_dim: int  # number of dimensions that describe the block world
    num_blocks: int  # number of blocks
    block_width: float = 1.0  # block width
    mode_connectivity: str = "sparse"
    # full -- allow transitioning into itself
    # sparse -- don't allow transitioning into itself
    add_time_cost: bool = True
    time_cost_weight: float = 1.0

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

    @property
    def delta(self) -> float:
        """Half block width"""
        return self.block_width / 2.0

    def __init__(self, block_dim: int = 1, num_blocks: int = 2, horizon: int = 5):
        self.block_dim = block_dim
        self.num_blocks = num_blocks
        self.horizon = horizon

        self.lb = np.zeros(self.state_dim)
        self.ub = 10.0 * np.ones(self.state_dim)

        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        self.graph_edges = np.empty((self.num_gcs_sets, self.num_gcs_sets))

        self.name_to_vertex = dict()

        self.modes_per_layer = []

    # display solution function: in text

    def set_block_width(self, block_width):
        self.block_width = block_width

    def set_ub(self, ub):
        self.ub = ub * np.ones(self.state_dim)

    ###################################################################################
    # Solve and display solution

    def solve(self, use_convex_relaxation=True, max_rounded_paths=30, show_graph=False):
        start_vertex = self.name_to_vertex["start"].id()
        target_vertex = self.name_to_vertex["target"].id()
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = use_convex_relaxation
        if use_convex_relaxation is True:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = max_rounded_paths
        INFO("Solving...")
        self.solution = self.gcs.SolveShortestPath(start_vertex, target_vertex, options)
        if self.solution.is_success():
            YAY("Optimal cost is %.1f" % self.solution.get_optimal_cost())
        else:
            ERROR("SOLVE FAILED!")
        if show_graph:
            self.display_graph()

    def display_graph(self) -> None:
        if self.solution.is_success():
            graphviz = self.gcs.GetGraphvizString(self.solution, True, precision=1)
        else:
            graphviz = self.gcs.GetGraphvizString()
        data = pydot.graph_from_dot_data(graphviz)[0]
        plt = Image(data.create_png())
        display(plt)

    def find_path_to_target(
        self,
        edges: T.List[GraphOfConvexSets.Edge],
        u: GraphOfConvexSets.Vertex,
    ) -> T.List[GraphOfConvexSets.Vertex]:
        # assuming edges are tight
        # find edge that has the current vertex as a start
        current_edge = next(e for e in edges if e.u() == u)
        # get the next vertex and continue
        v = current_edge.v()
        target_reached = v == self.name_to_vertex["target"]
        if target_reached:
            return [u] + [v]
        else:
            return [u] + self.find_path_to_target(edges, v)

    def get_solution_path(self) -> T.Tuple[T.List[str], npt.NDArray[np.float64]]:
        assert self.graph_built
        assert self.solution.is_success()
        # find edges with non-zero flow
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [self.solution.GetSolution(p) for p in flow_variables]
        active_edges = [
            edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow >= 0.99
        ]
        # using these edges, find the path from start to target
        path = self.find_path_to_target(active_edges, self.name_to_vertex["start"])
        modes = [v.name() for v in path]
        vertex_values = np.vstack([self.solution.GetSolution(v.x()) for v in path])
        return modes, vertex_values

    def verbose_solution_description(self) -> None:
        assert self.solution.is_success()
        modes, vertices = self.get_solution_path()
        print(vertices)
        for i in range(len(vertices)):
            vertices[i] = ["%.1f" % v for v in vertices[i]]
        mode_now = 0
        for i in range(len(modes)):
            sg = vertices[i][0 : self.block_dim]
            if modes[i] == "start":
                INFO("Start at", sg)
            elif modes[i] == "target":
                INFO("Move to", sg, "; Finish")
            else:
                mode_next = self.get_mode_from_name(modes[i])
                if mode_next == 0:
                    grasp = "Ungrasp block " + str(mode_now)
                else:
                    grasp = "Grasp   block " + str(mode_next)
                mode_now = mode_next
                INFO("Move to", sg, "; " + grasp)

    def get_mode_from_name(self, name: str) -> int:
        return int(name.split("_")[-1])

    ###################################################################################
    # Building the finite horizon GCS

    def populate_modes_per_layer(self) -> None:
        # at horizon 0 with have everything connected to initial_state
        initial_mode = 0
        self.modes_per_layer += [set(self.get_edges_out_of_set(initial_mode))]
        # for horizons 1 through h-1:
        for h in range(1, self.horizon):
            modes_at_next_layer = set()
            # for each modes at previous horizon
            for m in self.modes_per_layer[h - 1]:
                # add anything connected to it
                for k in self.get_edges_out_of_set(m):
                    modes_at_next_layer.add(k)
            self.modes_per_layer += [modes_at_next_layer]

    def build_the_graph(
        self,
        initial_state: Point,
        initial_set_id: int,
        final_state: Point,
        final_set_id: int,
    ) -> None:
        """
        Build the GCS graph of horizon H from start to target nodes.
        """
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False
        self.populate_edges_between_sets()
        self.populate_modes_per_layer()
        # populate the graph nodes layer by layer
        for layer in tqdm(range(self.horizon), desc="Adding layers: "):
            self.add_nodes_for_layer(layer)
        # add the start node
        self.add_start_node(initial_state, initial_set_id)
        # add the target node
        self.add_target_node(final_state, final_set_id)
        self.graph_built = True

    def add_nodes_for_layer(self, layer: int) -> None:
        """
        The GCS graph is a trellis diagram (finite horizon GCS); each layer contains an entire GCS graph of modes.
        Here we add the nodes, edges, constraints, and costs on the new layer.
        """
        for set_id in self.modes_per_layer[layer]:
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
        edges_in = self.modes_per_layer[self.horizon - 1]
        for left_vertex_set_id in edges_in:
            # add an edge
            left_vertex_name = self.get_vertex_name(
                self.horizon - 1, left_vertex_set_id
            )
            left_vertex = self.name_to_vertex[left_vertex_name]
            self.add_edge(left_vertex, target_vertex, left_vertex_set_id)

        if self.mode_connectivity == "sparse":
            # add edges between target and every node at mode 0
            for h in range(self.horizon - 1):
                if 0 in self.modes_per_layer[h]:
                    left_vertex_name = self.get_vertex_name(h, 0)
                    left_vertex = self.name_to_vertex[left_vertex_name]
                    self.add_edge(left_vertex, target_vertex, 0)

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
        self.add_constraints_on_edge(left_vertex_set_id, edge)
        # add movement cost on the edge
        self.add_gripper_movement_cost_on_edge(edge)
        # add time cost on edge
        if self.add_time_cost:
            self.add_time_cost_on_edge(edge)

    def add_vertex(self, state: HPolyhedron, name: str) -> GraphOfConvexSets.Vertex:
        # create a vertex
        vertex = self.gcs.AddVertex(state, name)
        self.name_to_vertex[name] = vertex
        # add a constraint on each vertex to be within lower/upper bound of the world
        # TODO: in theory, this should not be necessary
        set_con = LinearConstraint(np.eye(self.state_dim), self.lb, self.ub)
        vertex.AddConstraint(Binding[LinearConstraint](set_con, vertex.x()))
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

    def add_time_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        cost = L2NormCost(
            np.zeros((1, self.state_dim)), self.time_cost_weight * np.ones((1, 1))
        )
        edge.AddCost(Binding[Cost](cost, edge.xv()))

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

        eq_con = LinearEqualityConstraint(A, b)
        edge.AddConstraint(Binding[LinearEqualityConstraint](eq_con, np.append(xv, xu)))

        # add constraint that right point is in the left set
        left_vertex_set = self.get_convex_set_for_set_id(left_vertex_set_id)
        set_con = LinearConstraint(
            left_vertex_set.A(),
            -np.ones(left_vertex_set.b().size) * 1000,
            left_vertex_set.b(),
        )
        edge.AddConstraint(Binding[LinearConstraint](set_con, xv))

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

    def get_constraint_for_orbit_of_mode(
        self, mode: int
    ) -> T.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        A[0:bd, :] = np.zeros((bd, 2 * self.state_dim))
        # v_k - u_k is not constrained
        A[k * bd : (k + 1) * bd, :] = np.zeros((bd, 2 * self.state_dim))
        b = np.zeros(self.state_dim)
        return A, b

    ###################################################################################
    # Functions related to connectivity between modes and sets within modes

    def populate_edges_between_sets(self) -> None:
        """
        Return a matrix that represents edges in a directed graph of modes.
        For this simple example, the matrix is hand-built.
        When IRIS is used, sets must be A -- clustered (TODO: do they?),
        B -- connectivity checked and defined automatically.
        """
        assert self.mode_connectivity in ("full", "sparse")
        mat = np.zeros((self.num_gcs_sets, self.num_gcs_sets))
        if self.mode_connectivity == "full":
            # mode 0 is connected to any other mode
            mat[0, :] = np.ones(self.num_gcs_sets)
            # mode k is connected only to 0;
            mat[:, 0] = np.ones(self.num_gcs_sets)
        elif self.mode_connectivity == "sparse":
            # mode 0 is connected to any other mode except itself
            mat[0, 1:] = np.ones(self.num_gcs_sets - 1)
            # mode k is connected only to 0;
            mat[1:, 0] = np.ones(self.num_gcs_sets - 1)
        self.graph_edges = mat

    def get_edges_into_set(self, set_id: int) -> T.List[int]:
        """
        Use the edge matrix to determine which edges go into the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets
        return [v for v in range(self.num_gcs_sets) if self.graph_edges[v, set_id] == 1]

    def get_edges_out_of_set(self, set_id: int) -> T.List[int]:
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
