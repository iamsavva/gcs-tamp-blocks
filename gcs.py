import typing as T

import numpy as np
import numpy.typing as npt

import pydot
from tqdm import tqdm
from IPython.display import Image, display
import time

import pydrake.geometry.optimization as opt
from pydrake.geometry.optimization import Point, GraphOfConvexSets, HPolyhedron
from pydrake.solvers import (
    Binding,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearCost,
)

from util import ERROR, WARN, INFO, YAY


class GCSforBlocks:
    block_dim: int  # number of dimensions that describe the block world
    num_blocks: int  # number of blocks
    block_width: float = 1.0  # block width

    mode_connectivity: str = "sparse"
    # full -- allow transitioning into itself
    # sparse -- don't allow transitioning into itself

    # add a time cost on each edge? this is done to "regularize" the trajectory
    # goal is to reduce possibility of pointlessly grasping and ungrasping in place
    add_time_cost: bool = True
    time_cost_weight: float = 1.0  # relative weight between

    problem_complexity = "transparent-no-obstacles"

    ###################################################################################
    # Properties, inits, setter functions

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

    def set_block_width(self, block_width: float) -> None:
        self.block_width = block_width

    def set_ub(self, ub: float) -> None:
        self.ub = ub * np.ones(self.state_dim)

    def __init__(self, block_dim: int = 1, num_blocks: int = 2, horizon: int = 5):
        self.block_dim = block_dim
        self.num_blocks = num_blocks
        self.horizon = horizon

        # lower and upper bounds on the workspace
        self.lb = np.zeros(self.state_dim)
        self.ub = 10.0 * np.ones(self.state_dim)

        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        # name to vertex dictionary, populated as we populate the graph with vertices
        self.name_to_vertex = dict()  # T.Dict[str, GraphOfConvexSets.Vertex]

        # structures that hold information about the graph connectivity.
        # used for hand-built graph sparcity
        # see populate_important_things(self)
        self.sets_per_mode = dict()  # T.Dict[int, T.Set[int]]
        self.modes_per_layer = dict()  # T.Dict[int, T.Set[int]]
        self.sets_per_layer = dict()  # T.Dict[int, T.Set[int]]
        self.mode_graph_edges = np.empty([])  # np.NDArray, size num_modes x num_modes
        self.set_graph_edges = np.array(
            []
        )  # np.NDArray, size num_gcs_sets x num_gcs_sets

    ###################################################################################
    # Building the finite horizon GCS

    def build_the_graph(
        self,
        start_state: Point,
        start_mode: int,
        target_state: Point,
        target_mode: int,
    ) -> None:
        """
        READY
        Build the GCS graph of horizon H from start to target nodes.
        """
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        # hand-built pre-processing of the graph
        self.populate_important_things(start_mode)

        # add the start node and layer 0
        self.add_start_node(start_state, start_mode)

        # add layers 1 through horizon-1
        for layer in tqdm(range(1, self.horizon), desc="Adding layers: "):
            self.add_nodes_for_layer(layer)

        # add the target node
        self.add_target_node(target_state, target_mode)

        self.graph_built = True

    ###################################################################################
    # Adding layers of nodes (trellis diagram style)

    def add_nodes_for_layer(self, layer: int) -> None:
        """
        READY
        The GCS graph is a trellis diagram (finite horizon GCS); each layer contains an entire GCS graph of modes.
        Here we add the nodes, edges, constraints, and costs on the new layer.
        """
        assert 1 <= layer < self.horizon, "Invalid layer number"
        # for each mode in next layer
        for mode in self.modes_per_layer[layer]:
            # for each set in that mode, add new vertex
            for set_id in self.sets_per_mode[mode]:
                new_vertex = self.add_vertex(
                    self.get_convex_set_for_set_id(set_id),
                    self.get_vertex_name(layer, set_id),
                )
                # connect it with vertices from previouis layer
                edges_in = self.get_edges_into_set_out_of_mode(set_id)
                for left_vertex_set_id in edges_in:
                    # add an edge from previous layer
                    left_vertex_name = self.get_vertex_name(
                        layer - 1, left_vertex_set_id
                    )
                    left_vertex = self.name_to_vertex[left_vertex_name]
                    self.add_edge(left_vertex, new_vertex, left_vertex_set_id)

            # connect the vertices within the mode
            for left_vertex_set_id in self.sets_per_mode[mode]:
                connections_within_mode = self.get_edges_within_same_mode(
                    left_vertex_set_id
                )
                left_vertex_name = self.get_vertex_name(layer, left_vertex_set_id)
                left_vertex = self.name_to_vertex[left_vertex_name]
                # connect left vertex with other vertices within this mode
                for right_vertex_set_id in connections_within_mode:
                    right_vertex_name = self.get_vertex_name(layer, right_vertex_set_id)
                    right_vertex = self.name_to_vertex[right_vertex_name]
                    self.add_edge(left_vertex, right_vertex, left_vertex_set_id)

    def add_start_node(self, start_state: Point, start_mode: int) -> None:
        """
        READY
        Adds start node to the graph, as well as edges that connect to it.
        """
        # start is connected to each set in start_mode that contains it
        # horizon 0 contains only start_mode sets

        # add start node to the graph
        start_vertex = self.add_vertex(start_state, "start")
        # sets in start_mode
        sets_in_start_mode = self.sets_per_mode[start_mode]
        # add vertices into horizon 0
        for set_id in sets_in_start_mode:
            self.add_vertex(
                self.get_convex_set_for_set_id(set_id), self.get_vertex_name(0, set_id)
            )

        # obtain sets that contain the start point; these are the sets that start is connected to
        sets_with_start = []
        for set_in_mode in sets_in_start_mode:
            convex_set = self.get_convex_set_for_set_id(set_in_mode)
            if convex_set.PointInSet(start_state.x()):
                sets_with_start += [set_in_mode]
        assert (
            len(sets_with_start) > 0
        ), "No set in start mode contains the start point!"

        # add edges from start to the start-mode at horizon 0
        for set_id in sets_with_start:
            right_vertex_name = self.get_vertex_name(0, set_id)
            right_vertex = self.name_to_vertex[right_vertex_name]
            self.add_edge(start_vertex, right_vertex, set_id, False)

        # add edges within the start-mode at horizon 0
        for left_vertex_set_id in sets_in_start_mode:
            connections_within_mode = self.get_edges_within_same_mode(
                left_vertex_set_id
            )
            left_vertex_name = self.get_vertex_name(0, left_vertex_set_id)
            left_vertex = self.name_to_vertex[left_vertex_name]
            for right_vertex_set_id in connections_within_mode:
                right_vertex_name = self.get_vertex_name(0, right_vertex_set_id)
                right_vertex = self.name_to_vertex[right_vertex_name]
                self.add_edge(left_vertex, right_vertex, left_vertex_set_id)

    def add_target_node(self, target_state: Point, target_mode: int) -> None:
        """
        READY
        Adds target node to the graph, as well as edges that connect to it.
        """
        # add target vertex
        target_vertex = self.add_vertex(target_state, "target")

        # sets in target_mode
        sets_in_target_mode = self.sets_per_mode[target_mode]
        # sets that contain target point; these are the sets that can move into target in 1 step
        sets_with_target = []
        for set_in_mode in sets_in_target_mode:
            convex_set = self.get_convex_set_for_set_id(set_in_mode)
            if convex_set.PointInSet(target_state.x()):
                sets_with_target += [set_in_mode]
        assert (
            len(sets_with_target) > 0
        ), "No set in target mode contains the target point!"

        # at each horizon level, only sets that contain the target can transition into target
        # at each layer
        for layer in range(self.horizon):
            # if that layer has a target mode
            if target_mode in self.modes_per_layer[layer]:
                # for each set that contains the target
                for set_id in sets_with_target:
                    # this if statement shouldn't be necessary, but still
                    if set_id in self.sets_per_layer[layer]:
                        # add an edge
                        left_vertex_name = self.get_vertex_name(layer, set_id)
                        left_vertex = self.name_to_vertex[left_vertex_name]
                        self.add_edge(left_vertex, target_vertex, set_id)

    ###################################################################################
    # Populating edges and vertices

    def add_edge(
        self,
        left_vertex: GraphOfConvexSets.Vertex,
        right_vertex: GraphOfConvexSets.Vertex,
        left_vertex_set_id: int,
        add_set_transition_constraint=True,  # this setting exist to remove redundant constraints for out of start-mode
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
        # adding equality constraints
        self.add_orbital_constraint(left_vertex_set_id, edge)
        if add_set_transition_constraint:
            self.add_common_set_at_transition_constraint(left_vertex_set_id, edge)

        # -----------------------------------------------------------------
        # Adding costs
        # -----------------------------------------------------------------
        # add movement cost on the edge
        self.add_gripper_movement_cost_on_edge(edge)
        # add time cost on edge
        if self.add_time_cost:
            self.add_time_cost_on_edge(edge)

    def add_vertex(
        self, convex_set: HPolyhedron, name: str
    ) -> GraphOfConvexSets.Vertex:
        """
        NEEDS WORK, NOT TRANSPARENT
        Define a vertex with a convex set.
        Define upper/lower boundaries on variables to make CSDP solver happy.
        """
        # create a vertex
        # TODO: are there bounded polyhedrons?
        vertex = self.gcs.AddVertex(convex_set, name)
        self.name_to_vertex[name] = vertex
        # add a constraint on each vertex to be within lower/upper bound of the world
        # TODO: remove lb ub from convex set definition
        # set_con = LinearConstraint(np.eye(self.state_dim), self.lb, self.ub)
        # vertex.AddConstraint(Binding[LinearConstraint](set_con, vertex.x()))
        return vertex

    ###################################################################################
    # Adding constraints and cost terms
    def add_orbital_constraint(
        self, left_vertex_set_id: int, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        READY
        Add orbital constraints on the edge
        Orbital constraints are independent of edges
        """
        xu, xv = edge.xu(), edge.xv()
        left_mode = self.get_mode_from_set_id(left_vertex_set_id)
        A, b = self.get_constraint_for_orbit_of_mode(left_mode)

        eq_con = LinearEqualityConstraint(A, b)
        edge.AddConstraint(Binding[LinearEqualityConstraint](eq_con, np.append(xv, xu)))

    def add_common_set_at_transition_constraint(
        self, left_vertex_set_id: int, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        READY
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # get set that corresponds to left vertex
        left_vertex_set = self.get_convex_set_for_set_id(left_vertex_set_id)
        # fill in linear constraint on the right vertex
        A = left_vertex_set.A()
        lb = -np.ones(left_vertex_set.b().size) * 1000
        ub = left_vertex_set.b()
        set_con = LinearConstraint(A, lb, ub)
        edge.AddConstraint(Binding[LinearConstraint](set_con, edge.xv()))

    def add_gripper_movement_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        L2 norm cost on the movement of the gripper.
        """
        xu, xv = edge.xu(), edge.xv()
        #  gripper state is 0 to block_dim
        A = np.zeros((self.block_dim, 2 * self.state_dim))
        A[:, 0 : self.block_dim] = np.eye(self.block_dim)
        A[:, self.state_dim : self.state_dim + self.block_dim] = -np.eye(self.block_dim)
        b = np.zeros(self.block_dim)
        # add the cost
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[Cost](cost, np.append(xv, xu)))

    def add_time_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        Walking along the edges costs some cosntant term. This is done to avoid grasping and ungrasping in place.
        """
        cost = LinearCost(np.zeros(self.state_dim), self.time_cost_weight * np.ones(1))
        edge.AddCost(Binding[Cost](cost, edge.xv()))

    ###################################################################################
    # Generating convex sets per mode or per set in a mode

    def get_constraint_for_orbit_of_mode(
        self, mode: int
    ) -> T.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        READY
        Orbital constraint for blocks without pushing
        Delta x_i is zero for any i != 0, k; k is the mode number

        TODO: this should probably be implemented differently, for it implies constraint A(y-x) == b
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
    # IRIS
    def build_sets_per_mode(self) -> None:
        """
        NEEDS WORK
        Here I should be running IRIS to determine the sets for each mode.
        """
        assert self.problem_complexity in (
            "transparent-no-obstacles"
        ), "Non-transparent blocks not implemented yet"
        if self.problem_complexity == "transparent-no-obstacles":
            for mode in range(self.num_modes):
                self.sets_per_mode[mode] = {mode}

    def get_mode_from_set_id(self, set_id: int) -> int:
        """
        NEEDS WORK
        Returns a mode to which the vertex belongs.
        In the simple case of transparent blocks, each vertex represents the full mode, so it's just the vertex itself.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        assert self.problem_complexity in (
            "transparent-no-obstacles"
        ), "Non-transparent blocks not implemented yet"
        mode = set_id
        return mode

    def get_convex_set_for_set_id(self, set_id: int) -> HPolyhedron:
        """
        NEEDS WORK
        Returns convex set that corresponds to the given vertex.
        For the simple case of transparent blocks, this is equivivalent to the convex set that corresponds to the mode.
        """
        assert self.problem_complexity in (
            "transparent-no-obstacles"
        ), "Non-transparent blocks not implemented yet"
        if self.problem_complexity == "transparent-no-obstacles":
            return self.get_convex_set_for_mode_transparent(set_id)

    def get_convex_set_for_mode_transparent(self, mode: int) -> HPolyhedron:
        """
        READY
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

    ###################################################################################
    # Functions related to connectivity between modes and sets within modes

    def populate_important_things(self, initial_mode: int) -> None:
        """
        READY
        Preprocessing step that populates various graph-related things in an appropriate order.
        """
        # Run IRIS to determine sets that lie in individual modes
        self.build_sets_per_mode()
        # determine mode connectivity
        self.populate_edges_between_modes()
        # determine set connectivity
        self.populate_edges_between_sets()
        # determine which modes appear in which layer
        self.populate_modes_per_layer(initial_mode)
        # determine which sets appear in which layer
        self.populate_sets_per_layer()

    def populate_edges_between_modes(self) -> None:
        """
        READY
        Mode connectivity.
        """
        self.mode_graph_edges = np.zeros((self.num_modes, self.num_modes))
        # mode 0 is connected to any other mode except itself
        self.mode_graph_edges[0, 1:] = np.ones(self.num_modes - 1)
        # mode k is connected only to 0;
        self.mode_graph_edges[1:, 0] = np.ones(self.num_modes - 1)

    def populate_edges_between_sets(self) -> None:
        """
        NEEDS WORK
        Return a matrix that represents edges in a directed graph of modes.
        For this simple example, the matrix is hand-built.
        When IRIS is used, sets must be A -- clustered (TODO: do they?),
        B -- connectivity checked and defined automatically.
        """
        assert self.problem_complexity in (
            "transparent-no-obstacles"
        ), "Non-transparent blocks not implemented yet"
        if self.problem_complexity == "transparent-no-obstacles":
            assert self.mode_connectivity in (
                "sparse"
            ), "For transparent blocks must have sparse connectivity"
            self.set_graph_edges = np.zeros((self.num_gcs_sets, self.num_gcs_sets))
            # mode 0 is connected to any other mode except itself
            self.set_graph_edges[0, 1:] = np.ones(self.num_gcs_sets - 1)
            # mode k is connected only to 0;
            self.set_graph_edges[1:, 0] = np.ones(self.num_gcs_sets - 1)

    def populate_modes_per_layer(self, start_mode: int) -> None:
        """
        READY
        For each layer, determine what modes are reachable from the start node.
        """
        # at horizon 0 we have just the start mode
        self.modes_per_layer[0] = {start_mode}
        # for horizons 1 through h-1:
        for h in range(1, self.horizon):
            modes_at_next_layer = set()
            # for each modes at previous horizon
            for m in self.modes_per_layer[h - 1]:
                # add anything connected to it
                for k in self.get_edges_out_of_mode(m):
                    modes_at_next_layer.add(k)
            self.modes_per_layer[h] = modes_at_next_layer

    def populate_sets_per_layer(self) -> None:
        """
        READY
        If a mode belongs to a layer, all of its sets belong to a layer.
        """
        assert (
            len(self.modes_per_layer) == self.horizon
        ), "Must populate modes per layer first"
        # for each layer up to the horizon
        for layer in range(self.horizon):
            self.sets_per_layer[layer] = set()
            # for each mode in that layer
            for mode_in_layer in self.modes_per_layer[layer]:
                # for each set in that mode
                for set_in_mode in self.sets_per_mode[mode_in_layer]:
                    # add that set to the (set of sets) at that layer
                    self.sets_per_layer[layer].add(set_in_mode)

    ###################################################################################
    # Get edges in and out of a set

    def get_edges_into_set_out_of_mode(self, set_id: int) -> T.List[int]:
        """
        NEEDS WORK
        Use the edge matrix to determine which edges go into the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        return [
            v for v in range(self.num_gcs_sets) if self.set_graph_edges[v, set_id] == 1
        ]

    def get_edges_within_same_mode(self, set_id: int) -> T.List[int]:
        """
        NEEDS WORK
        edge matrix should contain only values for out of mode transitions
        here is where we deal with within-the-mode transitions
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        return []

    def get_edges_out_of_set_out_of_mode(self, set_id: int) -> T.List[int]:
        """
        NEEDS WORK
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        return [
            v for v in range(self.num_gcs_sets) if self.set_graph_edges[set_id, v] == 1
        ]

    def get_edges_out_of_mode(self, mode: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= mode < self.num_modes, "Mode out of bounds"
        return [v for v in range(self.num_modes) if self.mode_graph_edges[mode, v] == 1]

    ###################################################################################
    # Vertex and edge naming

    def get_vertex_name(self, layer: int, set_id: int) -> str:
        return "M_" + str(layer) + "_" + str(set_id)

    def get_mode_from_vertex_name(self, name: str) -> int:
        set_id = int(name.split("_")[-1])
        return self.get_mode_from_set_id(set_id)

    def get_edge_name(self, left_vertex_name: str, right_vertex_name: str) -> str:
        return "E: " + left_vertex_name + " -> " + right_vertex_name

    ###################################################################################
    # Solve and display solution

    def solve(self, use_convex_relaxation=True, max_rounded_paths=30, show_graph=False):
        """Solve the GCS program. Must build the graph first."""
        assert self.graph_built, "Must build graph first!"
        start_vertex = self.name_to_vertex["start"].id()
        target_vertex = self.name_to_vertex["target"].id()
        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = use_convex_relaxation
        if use_convex_relaxation is True:
            options.preprocessing = True  # TODO Do I need to deal with this?
            options.max_rounded_paths = max_rounded_paths
        INFO("Solving...")
        start = time.time()
        self.solution = self.gcs.SolveShortestPath(start_vertex, target_vertex, options)
        if self.solution.is_success():
            YAY("Solving GCS took %.2f seconds" % (time.time() - start))
            YAY("Optimal cost is %.1f" % self.solution.get_optimal_cost())
        else:
            ERROR("SOLVE FAILED!")
            ERROR("Solving GCS took %.2f seconds" % (time.time() - start))
        if show_graph:
            self.display_graph()

    def display_graph(self) -> None:
        """Use pydot to visually inspect the graph. If solution acquired -- also displays the solution."""
        assert self.graph_built, "Must build graph first!"
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
        start: GraphOfConvexSets.Vertex,
    ) -> T.List[GraphOfConvexSets.Vertex]:
        """Given a set of active edges, find a path from start to target"""
        # assuming edges are tight
        # find edge that has the current vertex as a start
        current_edge = next(e for e in edges if e.u() == start)
        # get the next vertex and continue
        v = current_edge.v()
        target_reached = v == self.name_to_vertex["target"]
        if target_reached:
            return [start] + [v]
        else:
            return [start] + self.find_path_to_target(edges, v)

    def get_solution_path(self) -> T.Tuple[T.List[str], npt.NDArray[np.float64]]:
        """Given a solved GCS problem, and assuming that it's tight, find a path from start to target"""
        assert self.graph_built, "Must build graph first!"
        assert self.solution.is_success(), "Solution was not found"
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
        """Describe the solution in text: grasp X, move to Y, ungrasp Z"""
        assert self.solution.is_success(), "Solution was not found"
        modes, vertices = self.get_solution_path()
        for i in range(len(vertices)):
            vertices[i] = ["%.1f" % v for v in vertices[i]]
        mode_now = self.get_mode_from_vertex_name(modes[1])
        INFO("-----------------------")
        INFO("Solution is:")
        INFO("-----------------------")
        for i in range(len(modes)):
            sg = vertices[i][0 : self.block_dim]
            if modes[i] == "start":
                INFO("Start at", sg)
            elif modes[i] == "target":
                INFO("Move to", sg, "; Finish")
            else:
                mode_next = self.get_mode_from_vertex_name(modes[i])
                if mode_next == mode_now:
                    grasp = ""
                elif mode_next == 0:
                    grasp = "Ungrasp block " + str(mode_now)
                else:
                    grasp = "Grasp   block " + str(mode_next)
                mode_now = mode_next
                INFO("Move to", sg, "; " + grasp)
