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

        self.lb = np.zeros(self.state_dim)
        self.ub = 10.0 * np.ones(self.state_dim)

        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        self.graph_edges = np.empty((self.num_gcs_sets, self.num_gcs_sets))

        self.name_to_vertex = dict()

        self.modes_per_layer = []

        self.sets_per_mode = dict()

    ###################################################################################
    # Building the finite horizon GCS

    def build_the_graph( 
        self,
        initial_state: Point,
        initial_mode: int,
        final_state: Point,
        final_mode: int,
    ) -> None:
        """
        NEEDS WORK
        Build the GCS graph of horizon H from start to target nodes.
        """
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False

        # TODO: need more populating here
        self.build_sets_per_mode()
        self.populate_edges_between_sets()
        self.populate_modes_per_layer()

        # populate the graph nodes layer by layer
        for layer in tqdm(range(self.horizon), desc="Adding layers: "):
            self.add_nodes_for_layer(layer)

        # add the start node
        initial_set_id = self.get_set_id_for_point_and_mode(initial_state, initial_mode)
        self.add_start_node(initial_state, initial_set_id)
        # add the target node
        final_set_id = self.get_set_id_for_point_and_mode(final_state, final_mode)
        self.add_target_node(final_state, final_set_id)
        
        self.graph_built = True

    

    ###################################################################################
    # Adding layers of nodes (trellis diagram style)

    def add_nodes_for_layer(self, layer: int) -> None: 
        """
        READY
        The GCS graph is a trellis diagram (finite horizon GCS); each layer contains an entire GCS graph of modes.
        Here we add the nodes, edges, constraints, and costs on the new layer.
        """
        # for each mode in next layer
        for mode in self.modes_per_layer[layer]:
            # for each set in that mode
            for set_id in self.sets_per_mode[mode]:
                # add new vertex
                new_vertex = self.add_vertex(
                    self.get_convex_set_for_set_id(set_id),
                    self.get_vertex_name(layer, set_id),
                )
                # edges and costs for layer 0 are populated in add_start_node
                if layer > 0:
                    edges_in = self.get_edges_into_set(set_id)
                    for left_vertex_set_id in edges_in:
                        # add an edge from previous layer
                        left_vertex_name = self.get_vertex_name(
                            layer - 1, left_vertex_set_id
                        )
                        left_vertex = self.name_to_vertex[left_vertex_name]
                        self.add_edge(left_vertex, new_vertex, left_vertex_set_id)

    def add_start_node(self, start_state: Point, start_set_id: int) -> None:
        """
        READY
        Adds start node to the graph.
        """
        # check that point belongs to the corresponding mode 
        # TODO: should be unnecessary 
        # convex_set = self.get_convex_set_for_set_id(start_set_id)
        # assert convex_set.PointInSet(start_state.x())
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
        NEEDS WORK
        Adds target node to the graph.
        """
        # check that point belongs to the corresponding mode
        # mode_set = self.get_convex_set_for_set_id(target_set_id)
        # assert mode_set.PointInSet(target_state.x())
        # add node to the graph
        target_vertex = self.add_vertex(target_state, "target")
        # TODO: do i need to look at set per layer?
        # get edges into target
        edges_in = self.modes_per_layer[self.horizon - 1]
        for left_vertex_set_id in edges_in: # TODO: must be sets
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
                    self.add_edge(left_vertex, target_vertex, 0) # TODO: need set id

    ###################################################################################
    # Populating edges and vertices

    def add_edge(
        self,
        left_vertex: GraphOfConvexSets.Vertex,
        right_vertex: GraphOfConvexSets.Vertex,
        left_vertex_set_id: int,
    ) -> None:
        """
        NEEDS WORK
        Add an edge between two vertices, as well as corresponding constraints and costs.
        """
        # add an edge
        edge_name = self.get_edge_name(left_vertex.name(), right_vertex.name())
        edge = self.gcs.AddEdge(left_vertex, right_vertex, edge_name)

        # -----------------------------------------------------------------
        # Adding constraints
        # -----------------------------------------------------------------
        self.add_orbital_constraint(left_vertex_set_id, edge)
        self.add_common_mode_at_transition_constraint(left_vertex_set_id, edge) # TODO: should be common set, not just common mode

        # -----------------------------------------------------------------
        # Adding costs
        # -----------------------------------------------------------------
        # add movement cost on the edge
        self.add_gripper_movement_cost_on_edge(edge)
        # add time cost on edge
        if self.add_time_cost:
            self.add_time_cost_on_edge(edge)

    def add_vertex(self, convex_set: HPolyhedron, name: str) -> GraphOfConvexSets.Vertex:
        """ 
        NEEDS WORK
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

    def add_common_mode_at_transition_constraint(
        self, left_vertex_set_id: int, edge: GraphOfConvexSets.Edge
    ) -> None:
        """
        NEEDS WORK
        Add a constraint that the right vertex belongs to the same mode as the left vertex
        """
        # TODO: should be common set
        left_vertex_set = self.get_convex_set_for_set_id(left_vertex_set_id)
        set_con = LinearConstraint(
            left_vertex_set.A(),
            -np.ones(left_vertex_set.b().size) * 1000,
            left_vertex_set.b(),
        )
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
        NEEDS WORK, NOT SETS
        Walking along the edges costs some cosntant term. This is done to avoid grasping and ungrasping in place.
        """
        # TODO: shouldn't be using an L2 cost, there are cleaner ways to do this
        cost = L2NormCost(
            np.zeros((1, self.state_dim)), self.time_cost_weight * np.ones((1, 1))
        )
        edge.AddCost(Binding[Cost](cost, edge.xv()))

    ###################################################################################
    # Generating convex sets per mode or per set in a mode

    def get_set_id_for_point_and_mode(self, state: Point, mode: int) -> int:
        """ 
        READY
        Given a mode and a point, find a set that belongs to the mode that includes that point.
        Note that there may be multiple sets that include this point; set with lowest index is returned.
        """
        sets_in_mode = self.sets_per_mode[mode]
        for set_id in sets_in_mode:
            convex_set = self.get_convex_set_for_set_id(set_id)
            if convex_set.PointInSet(state.x()):
                return set_id
        assert False, "Given point does not belong to a mode!"

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
        assert self.problem_complexity in ("transparent-no-obstacles"), "Non-transparent blocks not implemented yet"
        if self.problem_complexity == "transparent-no-obstacles":
            for m in range(self.num_modes):
                self.sets_per_mode[m] = [m]

    def get_convex_set_for_set_id(self, set_id: int) -> HPolyhedron:
        """
        NEEDS WORK
        Returns convex set that corresponds to the given vertex.
        For the simple case of transparent blocks, this is equivivalent to the convex set that corresponds to the mode.
        """
        assert self.problem_complexity in ("transparent-no-obstacles"), "Non-transparent blocks not implemented yet"
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

    def populate_modes_per_layer(self) -> None:
        """
        NEEDS WORK
        For each layer, determine what modes are reachable from the start node.
        """
        # at horizon 0 with have everything connected to initial_state
        # TODO: initial mode should be an input
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

    def populate_sets_per_layer(self) -> None:
        """
        NEEDS WORK
        """
        pass

    def populate_edges_between_sets(self) -> None:
        """
        NEEDS WORK
        Return a matrix that represents edges in a directed graph of modes.
        For this simple example, the matrix is hand-built.
        When IRIS is used, sets must be A -- clustered (TODO: do they?),
        B -- connectivity checked and defined automatically.
        """
        assert self.problem_complexity in ("transparent-no-obstacles"), "Non-transparent blocks not implemented yet"
        if self.problem_complexity == "transparent-no-obstacles":
            assert self.mode_connectivity in ("sparse"), "For transparent blocks must have sparse connectivity"
            mat = np.zeros((self.num_gcs_sets, self.num_gcs_sets))
            # mode 0 is connected to any other mode except itself
            mat[0, 1:] = np.ones(self.num_gcs_sets - 1)
            # mode k is connected only to 0;
            mat[1:, 0] = np.ones(self.num_gcs_sets - 1)
            self.graph_edges = mat

    def get_edges_into_set(self, set_id: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go into the vertex.
        """
        # NOTE: this must contain specifically edges from other modes
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        return [v for v in range(self.num_gcs_sets) if self.graph_edges[v, set_id] == 1]

    def get_edges_out_of_set(self, set_id: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        return [v for v in range(self.num_gcs_sets) if self.graph_edges[set_id, v] == 1]

    def get_mode_from_set_id(self, set_id: int) -> int:
        """
        NEEDS WORK
        Returns a mode to which the vertex belongs.
        In the simple case of transparent blocks, each vertex represents the full mode, so it's just the vertex itself.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        assert self.problem_complexity in ("transparent-no-obstacles"), "Non-transparent blocks not implemented yet"
        mode = set_id
        return mode

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
        mode_now = 0
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
                if mode_next == 0:
                    grasp = "Ungrasp block " + str(mode_now)
                else:
                    grasp = "Grasp   block " + str(mode_next)
                mode_now = mode_next
                INFO("Move to", sg, "; " + grasp)
