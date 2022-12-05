import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import le, eq
from .axis_aligned_set_tesselation_2d import (
    Box,
    axis_aligned_tesselation,
    locations_to_aligned_sets,
)
from .tsp_solver import Vertex, Edge
from .motion_planning_obstacles_on_off import MotionPlanning


class BlockMovingObstacleAvoidance:
    def __init__(
        self,
        start_pos,
        target_pos,
        bounding_box,
        block_width=1.0,
        convex_relaxation=False,
    ):
        self.num_blocks = len(start_pos) - 1
        assert len(target_pos) == len(start_pos)
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.vertices = dict()  # type: T.Dict[str, Vertex]
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.start_arm_pos = np.array(start_pos[0])
        self.target_arm_pos = np.array(target_pos[0])
        self.start_block_pos = [np.array(x) for x in start_pos[1:]]
        self.target_block_pos = [np.array(x) for x in target_pos[1:]]
        self.start = "sa_tsp"  # str
        self.target = "ta_tsp"  # str
        self.bounding_box = bounding_box
        # get obstacles
        obstacles = locations_to_aligned_sets(
            self.start_block_pos, self.target_block_pos, block_width, self.bounding_box
        )
        # make a tesselation
        self.convex_sets = axis_aligned_tesselation(bounding_box.copy(), obstacles)
        self.convex_relaxation = convex_relaxation
        self.prog = MathematicalProgram()
        self.solution = None
        self.build()

    @property
    def n(self):  # number of vertices
        return 2 * (self.num_blocks + 1)

    def s(self, name="a"):
        return "s" + str(name) + "_tsp"

    def t(self, name="a"):
        return "t" + str(name) + "_tsp"

    def add_vertex(self, name: str, value: npt.NDArray, block_index: int):
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = Vertex(name, value, block_index)

    def add_edge(self, left_name: str, right_name: str):
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists"
        self.edges[edge_name] = Edge(self.vertices[left_name], self.vertices[right_name], edge_name)
        self.vertices[left_name].add_edge_out(edge_name)
        self.vertices[right_name].add_edge_in(edge_name)

    def build(self):
        self.add_tsp_vertices_and_edges()
        self.add_tsp_variables_to_prog()
        self.add_tsp_constraints_to_prog()
        self.add_tsp_costs_to_prog()
        self.add_motion_planning()

    def add_tsp_vertices_and_edges(self):
        ################################
        # add all vertices
        ################################
        # add start/target arm vertices
        self.add_vertex(self.start, self.start_arm_pos, -1)
        self.add_vertex(self.target, self.target_arm_pos, -1)
        # add start/target block vertices
        for i, pos in enumerate(self.start_block_pos):
            self.add_vertex(self.s(i), pos, i)
        for i, pos in enumerate(self.target_block_pos):
            self.add_vertex(self.t(i), pos, i)

        ################################
        # add all edges
        ################################
        # add edge to from initial arm location to final arm location
        self.add_edge(self.s("a"), self.t("a"))
        for j in range(self.num_blocks):
            # from start to any
            self.add_edge(self.s("a"), self.s(j))
            # from any to target
            self.add_edge(self.t(j), self.t("a"))
            # from any to target to any start
            for i in range(self.num_blocks):
                if i != j:
                    self.add_edge(self.t(j), self.s(i))
            # from start to target is motion planning!

    def add_tsp_variables_to_prog(self):
        # each vertex has a visit variable
        # vertex visit serves as in for target, out for start
        for v in self.vertices.values():
            # visitation variable
            v.set_v(self.prog.NewContinuousVariables(self.num_blocks, "v_" + v.name))
            v.set_order(self.prog.NewContinuousVariables(1, "order_" + v.name)[0])

        for e in self.edges.values():
            # left and right visitation
            e.set_left_v(self.prog.NewContinuousVariables(self.num_blocks, "left_v_" + e.name))
            e.set_right_v(self.prog.NewContinuousVariables(self.num_blocks, "right_v_" + e.name))

            # add flow variable
            if self.convex_relaxation:
                e.set_phi(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
                self.prog.AddLinearConstraint(e.phi, 0.0, 1.0)
            else:
                e.set_phi(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])

            # left and right order
            e.set_left_order(self.prog.NewContinuousVariables(1, "left_order_" + e.name)[0])
            e.set_right_order(self.prog.NewContinuousVariables(1, "right_order" + e.name)[0])

    def add_tsp_constraints_to_prog(self):
        # for each edge, add constraints
        order_box = Box(lb=np.array([0]), ub=np.array([self.n - 1]), state_dim=1)
        visitation_box = Box(
            lb=np.zeros(self.num_blocks),
            ub=np.ones(self.num_blocks),
            state_dim=self.num_blocks,
        )
        for e in self.edges.values():
            # perspective constraints on order
            A, b = order_box.get_perspective_hpolyhedron()
            self.prog.AddLinearConstraint(le(A @ np.array([e.left_order, e.phi]), b))
            self.prog.AddLinearConstraint(le(A @ np.array([e.right_order, e.phi]), b))
            # perspective constraints on visits
            A, b = visitation_box.get_perspective_hpolyhedron()
            self.prog.AddLinearConstraint(le(A @ np.append(e.left_v, e.phi), b))
            self.prog.AddLinearConstraint(le(A @ np.append(e.right_v, e.phi), b))
            # increasing order
            self.prog.AddLinearConstraint(e.left_order + e.phi == e.right_order)
            # over all tsp edges, visit is same
            self.prog.AddLinearConstraint(eq(e.left_v, e.right_v))

        for v in self.vertices.values():
            flow_in = sum([self.edges[e].phi for e in v.edges_in])
            flow_out = sum([self.edges[e].phi for e in v.edges_out])
            order_in = sum([self.edges[e].right_order for e in v.edges_in])
            order_out = sum([self.edges[e].left_order for e in v.edges_out])
            v_in = sum([self.edges[e].right_v for e in v.edges_in])
            v_out = sum([self.edges[e].left_v for e in v.edges_out])
            if v.name == self.start:
                # it's the start vertex; initial conditions
                # flow out is 1
                self.prog.AddLinearConstraint(flow_out == 1)
                # order at vertex is 0
                self.prog.AddLinearConstraint(v.order == 0)
                # order continuity: order_out is 0
                self.prog.AddLinearConstraint(v.order == order_out)
                # 0 visits have been made yet
                self.prog.AddLinearConstraint(eq(v.v, np.zeros(self.num_blocks)))
                # visit continuity
                self.prog.AddLinearConstraint(eq(v.v, v_out))
            elif v.name == self.target:
                # it's the target vertex; final conditions
                # flow in is 1
                self.prog.AddLinearConstraint(flow_in == 1)
                # order at vertex is n-1
                self.prog.AddLinearConstraint(v.order == self.n - 1)
                # order continuity: order in is n-1
                self.prog.AddLinearConstraint(v.order == order_in)
                # all blocks have been visited
                self.prog.AddLinearConstraint(eq(v.v, np.ones(self.num_blocks)))
                # visit continuity: v me is v in
                self.prog.AddLinearConstraint(eq(v.v, v_in))
            elif v.name[0] == "s":
                # it's a start block vertex
                # flow in is 1
                self.prog.AddLinearConstraint(flow_in == 1)  # flow out is set in motion planning
                # vertex order is sum of orders in
                self.prog.AddLinearConstraint(v.order == order_in)
                # vertex visit is sum of visits in
                self.prog.AddLinearConstraint(eq(v.v, v_in))
                # order belongs to a set (TODO: redundant?)
                A, b = order_box.get_hpolyhedron()
                self.prog.AddLinearConstraint(le(A @ np.array([v.order]), b))
                # visitations belong to a set (TODO: redundant?)
                A, b = visitation_box.get_hpolyhedron()
                self.prog.AddLinearConstraint(le(A @ v.v, b))
            elif v.name[0] == "t":
                # it's a target block vertex
                # flow out is 1
                self.prog.AddLinearConstraint(flow_out == 1)  # flow in is set in motion planning
                # vertex order is sum of orders out
                self.prog.AddLinearConstraint(v.order == order_out)
                # vertex visit is sum of visits out
                self.prog.AddLinearConstraint(eq(v.v, v_out))
                # order belongs to a set (TODO: redundant?)
                A, b = order_box.get_hpolyhedron()
                self.prog.AddLinearConstraint(le(A @ np.array([v.order]), b))
                # visitations belong to a set (TODO: redundant?)
                A, b = visitation_box.get_hpolyhedron()
                self.prog.AddLinearConstraint(le(A @ v.v, b))

                # order / visitation continuity over motion planning
                # order at t_i_tsp = order at v_i_tsp + 1
                sv = self.vertices["s" + v.name[1:]]
                assert sv.block_index == v.block_index, "block indeces do not match"
                self.prog.AddLinearConstraint(sv.order + 1 == v.order)
                # visitations hold except for the block i at which we are in
                for i in range(self.num_blocks):
                    if i == v.block_index:
                        self.prog.AddLinearConstraint(sv.v[i] + 1 == v.v[i])
                    else:
                        self.prog.AddLinearConstraint(sv.v[i] == v.v[i])

    def add_tsp_costs_to_prog(self):
        for e in self.edges.values():
            e.cost = np.linalg.norm(e.right.value - e.left.value)
        self.prog.AddLinearCost(sum([e.phi * e.cost for e in self.edges.values()]))

    def add_motion_planning(self):
        for block_index in range(self.num_blocks):
            MotionPlanning(
                self.prog,
                self.vertices,
                self.edges,
                self.bounding_box.copy(),
                self.start_block_pos,
                self.target_block_pos,
                self.convex_sets,
                block_index,
                self.convex_relaxation,
            )

    def solve(self):
        x = timeit()
        self.solution = Solve(self.prog)
        x.dt("Solving the program")
        if self.solution.is_success():
            YAY("Optimal primal cost is %.5f" % self.solution.get_optimal_cost())
        else:
            ERROR("PRIMAL SOLVE FAILED!")
            ERROR("Optimal primal cost is %.5f" % self.solution.get_optimal_cost())
            raise Exception

        flows = [self.solution.GetSolution(e.phi) for e in self.edges.values()]
        not_tight = np.any(np.logical_and(0.01 < np.array(flows), np.array(flows) < 0.99))
        if not_tight:
            WARN("CONVEX RELAXATION NOT TIGHT")
        else:
            YAY("CONVEX RELAXATION IS TIGHT")

    def get_drawing_stuff(self):
        flow_vars = [(e, self.solution.GetSolution(e.phi)) for e in self.edges.values()]
        non_zero_edges = [e for (e, flow) in flow_vars if flow > 0.01]
        v_path, e_path = self.find_path_to_target(non_zero_edges, self.vertices[self.start])

        now_pose = self.start_pos.copy()
        now_mode = "start"
        poses = []
        modes = []

        def add_me(pose):
            p = pose.copy()
            p.resize(p.size)
            # if mode not in ("start", "target"):
            #     mode = str(int(mode)+1)
            poses.append(p)
            modes.append(0)

        i = 0
        while i < len(v_path):
            try:
                print(v_path[i].block_index, self.solution.GetSolution(v_path[i].v))
            except:
                pass
            if v_path[i].value is not None:
                now_pose[0] = v_path[i].value
            else:
                npq = self.solution.GetSolution(e_path[i].right_pos)
                now_pose[0] = npq
                now_pose[v_path[i].block_index + 1] = npq
            add_me(now_pose)
            i += 1
        return np.array(poses), modes

    def find_path_to_target(self, edges, start):
        """Given a set of active edges, find a path from start to target"""
        edges_out = [e for e in edges if e.left == start]
        assert len(edges_out) == 1
        current_edge = edges_out[0]
        v = current_edge.right

        target_reached = v.name == self.target

        if target_reached:
            return [start] + [v], [current_edge]
        else:
            v, e = self.find_path_to_target(edges, v)
            return [start] + v, [current_edge] + e

    # flow_vars = [(e, primal_solution.GetSolution(e.phi)) for e in edges.values()]
    # non_zero_edges = [e for (e, flow) in flow_vars if flow > 0.01]
    # v_path, e_path = find_path_to_target(non_zero_edges, vertices[start_tsp])
    # loc_path = [primal_solution.GetSolution(e.right_pos) for e in e_path]
    # loc_path[0] = primal_solution.GetSolution(e_path[1].left_pos)
