import typing as T

import numpy as np
import scipy
import numpy.typing as npt

import pydot
from tqdm import tqdm
from IPython.display import Image, display
import time

import pydrake.geometry.optimization as opt
from pydrake.geometry.optimization import Point, GraphOfConvexSets, HPolyhedron, Iris
from pydrake.solvers import (
    Binding,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearCost,
)
from pydrake.common import RandomGenerator

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

    # problem_complexity = "transparent-no-obstacles"
    problem_complexity = "obstacles"

    no_cycles = False

    ###################################################################################
    # Properties, inits, setter functions

    @property
    def num_modes(self) -> int:
        """
        Number of modes. For the case with no pushing, we have 1 mode for free motion and a mode per block for when grasping that block.
        The case with pushing will have many more modes.
        """
        return self.num_blocks + 1

    # @property
    # def num_gcs_sets(self) -> int:
    #     """
    #     Returns number of GCS sets; right now it's just the number of modes, but in general each mode will have multiple convex sets as part of it.
    #     """
    #     return self.num_modes * 1

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
        self.set_id_to_polyhedron = dict() # T.Dict[int, HPolyhedron]
        self.num_gcs_sets = -1

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
                if not self.no_cycles:
                    # connect it with vertices from previouis layer
                    edges_in = self.get_edges_into_set_out_of_mode(set_id)
                    for left_vertex_set_id in edges_in:
                        # add an edge from previous layer
                        left_vertex_name = self.get_vertex_name(
                            layer - 1, left_vertex_set_id
                        )
                        left_vertex = self.name_to_vertex[left_vertex_name]
                        self.add_edge(left_vertex, new_vertex, left_vertex_set_id)
                else:
                    # connect it with vertices from previouis layer
                    edges_in = self.get_edges_into_set(set_id)
                    for left_vertex_set_id in edges_in:
                        # add an edge from previous layer
                        left_vertex_name = self.get_vertex_name(
                            layer - 1, left_vertex_set_id
                        )
                        try:
                            left_vertex = self.name_to_vertex[left_vertex_name]
                            self.add_edge(left_vertex, new_vertex, left_vertex_set_id)
                        except:
                            pass
            
            if not self.no_cycles:
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
                        self.add_edge(
                            left_vertex,
                            right_vertex,
                            left_vertex_set_id,
                            add_grasp_cost=False,
                        )

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
            self.add_edge(
                start_vertex,
                right_vertex,
                set_id,
                add_set_transition_constraint=False,
                add_grasp_cost=False,
            )
        if not self.no_cycles:
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
                    self.add_edge(
                        left_vertex, right_vertex, left_vertex_set_id, add_grasp_cost=False
                    )

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
                        self.add_edge(
                            left_vertex, target_vertex, set_id, add_grasp_cost=False
                        )

    ###################################################################################
    # Populating edges and vertices

    def add_edge(
        self,
        left_vertex: GraphOfConvexSets.Vertex,
        right_vertex: GraphOfConvexSets.Vertex,
        left_vertex_set_id: int,
        add_set_transition_constraint: bool = True,  # this setting exist to remove redundant constraints for out of start-mode
        add_grasp_cost: bool = True,
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
        if add_grasp_cost and self.add_time_cost:
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
        vertex = self.gcs.AddVertex(convex_set, name)
        self.name_to_vertex[name] = vertex
        # TODO: put this check at set generation
        if not convex_set.IsBounded():
            WARN("Convex set", name, "is not bounded!")
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
        orbital_constraint = self.get_orbital_constraint(left_mode)
        edge.AddConstraint(
            Binding[LinearEqualityConstraint](orbital_constraint, np.append(xv, xu))
        )

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
        d = self.block_dim
        n = self.state_dim
        A = np.zeros((d, 2 * n))
        A[:, 0:d] = np.eye(d)
        A[:, n : n + d] = -np.eye(d)
        b = np.zeros(d)
        # add the cost
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    def add_time_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        Walking along the edges costs some cosntant term. This is done to avoid grasping and ungrasping in place.
        """
        A = np.zeros(self.state_dim)
        b = self.time_cost_weight * np.ones(1)
        cost = LinearCost(A, b)
        edge.AddCost(Binding[LinearCost](cost, edge.xv()))

    ###################################################################################
    # Orbital sets and constraints

    def get_orbit_set_for_mode_equality(self, mode: int):
        """
        When in mode k, the orbit is such that x_m-y_m = 0 for m not k nor 0.
        Produces convex set in a form A [x, y]^T = b
        """
        A = None
        b = None
        d = self.block_dim
        n = self.state_dim
        # for each mode
        for m in range(self.num_modes):
            # that is not 0 or mode
            if m not in (0, mode):
                # add constraint
                A_m = np.zeros((d, 2 * n))
                A_m[:, d * m : d * (m + 1)] = np.eye(d)
                A_m[:, n + d * m : n + d * (m + 1)] = -np.eye(d)
                b_m = np.zeros(d)
                if A is None:
                    A = A_m
                    b = b_m
                else:
                    A = np.vstack((A, A_m))
                    b = np.hstack((b, b_m))
        return A, b

    def get_orbit_set_for_mode_inequality(self, mode: int):
        """
        When in mode k, the orbit is such that x_m-y_m = 0 for m not k nor 0.
        Produces convex set in a form A [x, y]^T <= b
        """
        A, b = self.get_orbit_set_for_mode_equality(mode)
        return self.get_inequality_form_from_equality_form(A, b)

    def get_orbital_constraint(self, mode: int):
        A, b = self.get_orbit_set_for_mode_equality(mode)
        return LinearEqualityConstraint(A, b)

    ###################################################################################
    # Trivial representation transformations

    def get_inequality_form_from_equality_form(
        self, A: npt.NDArray, b: npt.NDArray
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Given a set in a form Ax = b return this same set in a form Ax <= b
        """
        new_A = np.vstack((A, -A))
        new_b = np.hstack((b, -b))
        return new_A, new_b

    ###################################################################################
    # Sets for modes, done clean

    def get_bounding_box_on_x_two_inequalities(
        self,
    ) -> T.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Bounding box on x is lb <= x <= ub.
        Returns this inequality in a form lb <= Ax <= ub.
        """
        A = np.eye(self.state_dim)
        lb = self.lb
        ub = self.ub
        return A, lb, ub

    def get_bounding_box_on_x_single_inequality(
        self,
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Bounding box on x is lb <= x <= ub.
        Returns this inequality in a form Ax <= b
        """
        A, lb, ub = self.get_bounding_box_on_x_two_inequalities()
        AA = np.vstack((A, -A))
        b = np.hstack((ub, -lb))
        return AA, b

    def get_plane_for_grasping_modes_equality(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        When gasping block m, x_0 = x_m. The plane of possible states when in mode k is given by
        x_0 - x_k = 0.
        Returns this plane in the form Ax = b.
        """
        d = self.block_dim
        n = self.state_dim
        A = np.zeros((d, n))
        A[0:d, 0:d] = np.eye(d)
        A[0:d, mode * d : (mode + 1) * d] = -np.eye(d)
        b = np.zeros(d)
        return A, b

    def get_plane_for_grasping_modes_inequality(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        When gasping block m, x_0 = x_m. The plane of possible states when in mode k is given by
        x_0 - x_k = 0.
        Returns this plane in the form Ax <= b.
        """
        A, b = self.get_plane_for_grasping_modes_equality(mode)
        return self.get_inequality_form_from_equality_form(A, b)

    def get_convex_set_for_mode_inequality(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Convex set for mode 0 is just the bounding box.
        Convex set for mode k is the bounding box and a plane.
        Returns a convex set for mode in form Ax <= b.
        """
        if mode == 0:
            return self.get_bounding_box_on_x_single_inequality()
        else:
            A_bounding, b_bounding = self.get_bounding_box_on_x_single_inequality()
            A_plane, b_plane = self.get_plane_for_grasping_modes_inequality(mode)
            A = np.vstack((A_bounding, A_plane))
            b = np.hstack((b_bounding, b_plane))
            return A, b

    def get_convex_set_for_mode_polyhedron(self, mode: int) -> HPolyhedron:
        """See get_convex_set_for_mode_inequality"""
        A, b = self.get_convex_set_for_mode_inequality(mode)
        return HPolyhedron(A, b)

    ###################################################################################
    # Obstacles
    def obstacle_in_configuration_space_inequality(
        self, block: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        When in mode 0, there are no obstacles.
        When in mode m, block m cannot collide with other blocks. Other block is given as an obstacle:
        |x_block - x_m| <= block_width
        Since x_m = x_0 in mode k, we have:
        |x_block - x_0| <= block_width

        Returns this obstacle in configuration space as an inequality Ax<=b
        """
        # TODO: should I also add constraint on mode? so both on 0 and on mode?
        # TODO: should I add a boundary?
        d = self.block_dim
        n = self.state_dim
        A = np.zeros((d, n))
        A[:, 0:d] = np.eye(d)
        A[:, block * d : (block + 1) * d] = -np.eye(d)
        b = np.ones(d) * self.block_width
        A = np.vstack((A, -A))
        b = np.hstack((b, b))
        return A, b

    def obstacle_in_configuration_space_polyhedron(self, block: int) -> HPolyhedron:
        """See obstacle_in_configuration_space_inequality"""
        A, b = self.obstacle_in_configuration_space_inequality(block)
        return HPolyhedron(A, b)

    ###################################################################################
    # Mode space transformation

    def transformation_between_configuration_and_mode_space(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Contact-with-block modes are planes in R^n: Ax=b.
        Instead of operating in a n-dimensional space, we can operate on an affine space that is a nullspace of A:
        for x_0 s.t. Ax_0 = b and N = matrix of vectors of the nullspace of A, we have:
        any x, s.t. Ax=b is given by x = x_0 + Ny, where y is of dimension of the nullspace of A.
        This function returns some pair x_0 and N.
        """
        A, b = self.get_plane_for_grasping_modes_equality(mode)
        x_0, residuals = np.linalg.lstsq(A, b, rcond=None)[0:2]
        # print(x_0, residuals)
        # assert np.allclose(residuals, np.zeros(self.state_dim)), "Residuals non zero when solving Ax=b"
        N = scipy.linalg.null_space(A)
        return x_0, N

    def transformation_between_mode_and_configuration_space(
        self, mode: int
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        We can move from mode space into the configuration space using pseudo inverse
        """
        x_0, N = self.transformation_between_configuration_and_mode_space(mode)
        mpi = np.linalg.pinv(N)
        return x_0, mpi

    def configuration_space_inequality_in_mode_space_inequality(
        self, mode: int, A: npt.NDArray, b: npt.NDArray
    ) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """
        Suppose a polyhedron in configuration space is given by Ax <= b
        The mode space for mode is x = x_0 + Ny
        Plugging in, we have obstacle in mode space:
        Ax_0 + ANy <= b
        ANy <= b-Ax_0
        returns AN, b-Ax_0, which define the obstacle in mode space
        """
        # get transformation into mode space
        x_0, N = self.transformation_between_configuration_and_mode_space(mode)
        return A.dot(N), (b - A.dot(x_0))

    def configuration_space_obstacle_in_mode_space(
        self, mode: int, block: int
    ) -> HPolyhedron:
        """
        See inequality_polyhedron_in_mode_space_inequality.
        """
        # get obstacle
        A, b = self.obstacle_in_configuration_space_inequality(block)
        A_m, b_m = self.configuration_space_inequality_in_mode_space_inequality(
            mode, A, b
        )
        return HPolyhedron(A_m, b_m)

    def mode_space_polyhedron_in_configuration_space(
        self, mode: int, poly: HPolyhedron
    ) -> HPolyhedron:
        """
        we can transform polyhedrons in configuration space into polyhedrons in mode space
        """
        A, b = poly.A(), poly.b()
        x_0, mpi = self.transformation_between_mode_and_configuration_space(mode)
        A_c = A.dot(mpi)
        b_c = b + A.dot(mpi.dot(x_0))
        return HPolyhedron(A_c, b_c)

    ###################################################################################
    # Running IRIS

    def get_convex_tesselation_for_mode(self, mode: int) -> T.List[HPolyhedron]:
        """
        NEEDS TESTING
        """

        def combine_sets(A_1:npt.NDArray, A_2:npt.NDArray, b_1:npt.NDArray, b_2:npt.NDArray):
            A, b = np.vstack((A_1, A_2)), np.hstack((b_1, b_2))
            return HPolyhedron(A, b)

        # get mode space obstacles
        obstacle_blocks = [i for i in range(1, self.num_modes) if i != mode]
        mode_space_obstacles = [
            self.configuration_space_obstacle_in_mode_space(mode, block)
            for block in obstacle_blocks
        ]
        # get mode space domain
        (
            conf_space_dom_A,
            conf_space_dom_b,
        ) = self.get_bounding_box_on_x_single_inequality()
        # YAY(conf_space_dom_A, conf_space_dom_b)
        (
            mode_space_dom_A,
            mode_space_dom_b,
        ) = self.configuration_space_inequality_in_mode_space_inequality(
            mode, conf_space_dom_A, conf_space_dom_b
        )

        # ERROR(mode_space_dom_A, mode_space_dom_b)

        mode_space_domain = HPolyhedron(mode_space_dom_A, mode_space_dom_b)
        mode_space_domain = mode_space_domain.ReduceInequalities()
        # ERROR(mode_space_domain.A(), mode_space_domain.b())

        # get IRIS tesselation
        mode_space_tesselation = self.get_IRIS_tesselation(
            mode_space_obstacles, mode_space_domain
        )
        # move IRIS tesselation into configuration space
        configuration_space_tesselation = [
            self.mode_space_polyhedron_in_configuration_space(mode, poly)
            for poly in mode_space_tesselation
        ]
        # add in-mode constraint to each polyhedron (TODO: this should be redundant?)
        A_mode, b_mode = self.get_convex_set_for_mode_inequality(mode)
        convex_sets_for_mode = [
            combine_sets(A_mode, c.A(), b_mode, c.b())
            for c in configuration_space_tesselation
        ]
        INFO("Iris finished mode", mode)
        return convex_sets_for_mode

    def get_IRIS_tesselation(
        self,
        obstacles: T.List[HPolyhedron],
        domain: HPolyhedron,
        max_num_sets: int = 9,
        max_num_samples: int = 100,
    ) -> T.List[HPolyhedron]:
        """
        NEEDS TESTING
        """
        sample_counter = 0
        previous_sample = None
        convex_sets = []
        generator = RandomGenerator()
        while sample_counter < max_num_samples:
            # sample a point
            sample_counter += 1
            if previous_sample is None:
                new_sample = domain.UniformSample(generator)
            else:
                new_sample = domain.UniformSample(generator, previous_sample=previous_sample)
            previous_sample = new_sample
            # check that a sampled point is not in any obstacle or in already attained set
            sample_not_inside_obstacle_or_existing_sets = True
            for some_set in obstacles + convex_sets:
                if some_set.PointInSet(new_sample):
                    sample_not_inside_obstacle_or_existing_sets = False
                    break
            if not sample_not_inside_obstacle_or_existing_sets:
                continue
            convex_set = Iris(obstacles, new_sample, domain)

            convex_sets.append(convex_set)
            if len(convex_sets) == max_num_sets:
                INFO("found max number of convex sets")
                return convex_sets
        INFO("sampled many points")
        return convex_sets

    ###################################################################################
    # build sets that are inside the modes

    def build_sets_per_mode(self) -> None:
        """
        READY
        Here I should be running IRIS to determine the sets for each mode.
        """
        assert self.problem_complexity in (
            "transparent-no-obstacles",
            "obstacles"
        ), "Problem complexity option not implemented"
        if self.problem_complexity == "transparent-no-obstacles":
            for mode in range(self.num_modes):
                self.sets_per_mode[mode] = {mode}
            self.num_gcs_sets = self.num_modes

        elif self.problem_complexity == "obstacles":
            # mode 0 is collision free and hence has just a single set in it
            self.sets_per_mode[0] = {0}
            self.set_id_to_polyhedron[0] = self.get_convex_set_for_mode_polyhedron(0)
            set_indexer = 1
            # for each mode
            for mode in range(1, self.num_modes):
                # get convex sets that belong to this mode
                sets_in_mode = self.get_convex_tesselation_for_mode(mode)
                self.sets_per_mode[mode] = set()
                print("mode ", mode, "has convex sets:", len(sets_in_mode))
                for some_set in sets_in_mode:
                    some_set = some_set.ReduceInequalities()
                    self.sets_per_mode[mode].add(set_indexer)
                    self.set_id_to_polyhedron[set_indexer] = some_set
                    set_indexer += 1
            self.num_gcs_sets = set_indexer

    def get_mode_from_set_id(self, set_id: int) -> int:
        """
        READY
        Returns a mode to which the vertex belongs.
        In the simple case of transparent blocks, each vertex represents the full mode, so it's just the vertex itself.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        assert self.problem_complexity in (
            "transparent-no-obstacles", "obstacles"
        ), "Problem complexity option not implemented"
        if self.problem_complexity == "transparent-no-obstacles":
            mode = set_id
            return mode
        elif self.problem_complexity == "obstacles":
            for mode in range(self.num_modes):
                if set_id in self.sets_per_mode[mode]:
                    return mode
            assert False, "Set_id not in any mode??"
        raise NotImplementedError

    def get_convex_set_for_set_id(self, set_id: int) -> HPolyhedron:
        """
        READY
        Returns convex set that corresponds to the given vertex.
        For the simple case of transparent blocks, this is equivivalent to the convex set that corresponds to the mode.
        """
        assert self.problem_complexity in (
            "transparent-no-obstacles",
            "obstacles"
        ), "Problem complexity option not implemented"
        if self.problem_complexity == "transparent-no-obstacles":
            mode = set_id
            return self.get_convex_set_for_mode_polyhedron(mode)
        elif self.problem_complexity == "obstacles":
            return self.set_id_to_polyhedron[set_id]
        raise NotImplementedError

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
        if self.no_cycles:
            # mode 0 is connected to any other mode except itself
            self.mode_graph_edges[0, :] = np.ones(self.num_modes)
            # mode k is connected only to 0;
            self.mode_graph_edges[:, 0] = np.ones(self.num_modes)
        else:
            # mode 0 is connected to any other mode except itself
            self.mode_graph_edges[0, 1:] = np.ones(self.num_modes - 1)
            # mode k is connected only to 0;
            self.mode_graph_edges[1:, 0] = np.ones(self.num_modes - 1)

    def populate_edges_between_sets(self) -> None:
        """
        READY
        Return a matrix that represents edges in a directed graph of modes.
        For this simple example, the matrix is hand-built.
        When IRIS is used, sets must be A -- clustered (TODO: do they?),
        B -- connectivity checked and defined automatically.
        """
        assert self.problem_complexity in (
            "transparent-no-obstacles",
            "obstacles"
        ), "Problem complexity option not implemented"

        if self.problem_complexity == "transparent-no-obstacles":
            assert self.mode_connectivity in (
                "sparse"
            ), "For transparent blocks must have sparse connectivity"
            self.set_graph_edges = np.zeros((self.num_gcs_sets, self.num_gcs_sets))
            # mode 0 is connected to any other mode except itself
            self.set_graph_edges[0, 1:] = np.ones(self.num_gcs_sets - 1)
            # mode k is connected only to 0;
            self.set_graph_edges[1:, 0] = np.ones(self.num_gcs_sets - 1)

        elif self.problem_complexity == "obstacles":
            self.set_graph_edges = np.zeros((self.num_gcs_sets, self.num_gcs_sets))
            # TODO: this a slight hack, technically needs better implementation
            # TODO: specifically, this is because i directly use the knowledge that 0 to everything everythin to 0
            # for each set in mode 0

            # connnect 0 to other modes
            for i in self.sets_per_mode[0]:
                poly_i = self.set_id_to_polyhedron[i]
                # for each other set
                for j in range(i+1, self.num_gcs_sets):
                    poly_j = self.set_id_to_polyhedron[j]
                    # print(poly_i, poly_j)
                    # if they intersect -- add an edge
                    if poly_i.IntersectsWith(poly_j):
                        self.set_graph_edges[i,j] = 1
                        self.set_graph_edges[j,i] = 1

            # connect other modes with themselves
            for mode in range(1, self.num_modes):
                sets_in_mode = self.sets_per_mode[mode]
                for set_id in sets_in_mode:
                    convex_set = self.get_convex_set_for_set_id(set_id)
                    for other_set_id in sets_in_mode:
                        if set_id != other_set_id:
                            other_convex_set = self.get_convex_set_for_set_id(other_set_id)
                            if convex_set.IntersectsWith(other_convex_set):
                                self.set_graph_edges[set_id, other_set_id] = 1
                                self.set_graph_edges[other_set_id, set_id] = 1



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
        READY
        Use the edge matrix to determine which edges go into the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        if self.problem_complexity == "transparent-no-obstacles":
            return [
                v for v in range(self.num_gcs_sets) if self.set_graph_edges[v, set_id] == 1
            ]
        elif self.problem_complexity == "obstacles":
            edges = []
            mode = self.get_mode_from_set_id(set_id)
            # for each mode that enters into our mode
            modes_into_me = self.get_edges_into_mode(mode)
            for other_mode in modes_into_me:
                if not self.no_cycles:
                    assert other_mode != mode
                # for each set in that other mode
                for i in self.sets_per_mode[other_mode]:
                    # check if there is an edge
                    if self.set_graph_edges[i, set_id] == 1:
                        edges.append(i)
            return edges
        raise NotImplementedError
            
    def get_edges_within_same_mode(self, set_id: int) -> T.List[int]:
        """
        READY
        edge matrix should contain only values for out of mode transitions
        here is where we deal with within-the-mode transitions
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        if self.problem_complexity == "transparent-no-obstacles":
            return []
        elif self.problem_complexity == "obstacles":
            mode = self.get_mode_from_set_id(set_id)
            edges = []
            for i in self.sets_per_mode[mode]:
                if i != set_id and self.set_graph_edges[set_id, i] == 1:
                    edges.append(i)
            return edges
        raise NotImplementedError

    def get_edges_out_of_set_out_of_mode(self, set_id: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= set_id < self.num_gcs_sets, "Set number out of bounds"
        if self.problem_complexity == "transparent-no-obstacles":
            return [
                v for v in range(self.num_gcs_sets) if self.set_graph_edges[set_id, v] == 1
            ]
        elif self.problem_complexity == "obstacles":
            edges = []
            mode = self.get_mode_from_set_id(set_id)
            # for each mode that we go into
            modes_out_of_me = self.get_edges_out_of_mode(mode)
            for other_mode in modes_out_of_me:
                assert other_mode != mode
                # for each set in that other mode
                for i in self.sets_per_mode[other_mode]:
                    # check if there is an edge
                    if self.set_graph_edges[set_id, i] == 1:
                        edges.append(i)
            return edges
        raise NotImplementedError

    def get_edges_into_set(self, set_id):
        return self.get_edges_into_set_out_of_mode(set_id) + self.get_edges_within_same_mode(set_id)
        
    def get_edges_out_of_set(self, set_id):
        return self.get_edges_out_of_set_out_of_mode(set_id) + self.get_edges_within_same_mode(set_id)

    def get_edges_out_of_mode(self, mode: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= mode < self.num_modes, "Mode out of bounds"
        return [v for v in range(self.num_modes) if self.mode_graph_edges[mode, v] == 1]

    def get_edges_into_mode(self, mode: int) -> T.List[int]:
        """
        READY
        Use the edge matrix to determine which edges go out of the vertex.
        """
        assert 0 <= mode < self.num_modes, "Mode out of bounds"
        return [v for v in range(self.num_modes) if self.mode_graph_edges[v, mode] == 1]

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
        modes = [
            str(self.get_mode_from_vertex_name(mode))
            if mode not in ("start", "target")
            else mode
            for mode in modes
        ]
        vertex_values = np.vstack([self.solution.GetSolution(v.x()) for v in path])
        return modes, vertex_values

    def verbose_solution_description(self) -> None:
        """Describe the solution in text: grasp X, move to Y, ungrasp Z"""
        assert self.solution.is_success(), "Solution was not found"
        modes, vertices = self.get_solution_path()
        for i in range(len(vertices)):
            vertices[i] = ["%.1f" % v for v in vertices[i]]
        mode_now = modes[1]
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
                mode_next = modes[i]
                if mode_next == mode_now:
                    grasp = ""
                elif mode_next == "0":
                    grasp = "Ungrasp block " + str(mode_now)
                else:
                    grasp = "Grasp block " + str(mode_next)
                mode_now = mode_next
                INFO("Move to", sg, "; " + grasp)
