import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import le, eq
from .axis_aligned_set_tesselation_2d import Box

# import graphviz


class Vertex:
    def __init__(self, name: str, value = None, block_index=None):
        self.value = value  # effectively just the name
        self.name = name  # name of the vertex
        self.edges_in = []  # str names of edges in
        self.edges_out = []  # str names of edges out
        self.block_index = block_index

        self.v = None
        self.order = None

    def set_block_index(self, block_index: int):
        assert self.block_index is None, (
            "Block index for " + self.name + " is already set"
        )
        self.block_index = block_index

    def add_edge_in(self, nbh: str):
        assert nbh not in self.edges_in
        self.edges_in.append(nbh)

    def add_edge_out(self, nbh: str):
        assert nbh not in self.edges_out
        self.edges_out.append(nbh)

    def set_v(self, v):
        assert self.v is None, "V for " + self.name + " is already set"
        self.v = v

    def set_order(self, order):
        assert self.order is None, "Order for " + self.name + " is already set"
        self.order = order


class Edge:
    def __init__(
        self, left_vertex: Vertex, right_vertex: Vertex, name: str, cost: float = None
    ):
        self.left = left_vertex
        self.right = right_vertex
        self.name = name
        self.cost = cost

        # primal variables
        self.phi = 0
        self.left_pos = 0
        self.right_pos = 0

        self.left_order = 0
        self.right_order = 0

        self.left_v = 0
        self.right_v = 0

    def set_cost(self, cost: float):
        assert self.cost is None, "Cost for " + self.name + " is already set"
        self.cost = cost

    def set_phi(self, flow):
        assert self.phi == 0, "Flow for " + self.name + " is already set"
        self.phi = flow

    def set_left_pos(self, left_pos):
        assert self.left_pos == 0, "left_pos for " + self.name + " is already set"
        self.left_pos = left_pos

    def set_right_pos(self, right_pos):
        assert self.right_pos == 0, "right_pos for " + self.name + " is already set"
        self.right_pos = right_pos

    def set_left_order(self, left_order):
        assert self.left_order == 0, "left_order for " + self.name + " is already set"
        self.left_order = left_order

    def set_right_order(self, right_order):
        assert self.right_order == 0, "right_order for " + self.name + " is already set"
        self.right_order = right_order

    def set_left_v(self, left_v):
        assert self.left_v == 0, "left_v for " + self.name + " is already set"
        self.left_v = left_v

    def set_right_v(self, right_v):
        assert self.right_v == 0, "right_v for " + self.name + " is already set"
        self.right_v = right_v


class TSPasGCS:
    def __init__(self):
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.vertices = dict()  # type: T.Dict[str, Vertex]
        self.start = None  # str
        self.target = None  # str
        self.primal_prog = None  # MathematicalProgram
        self.primal_solution = None

    @property
    def n(self):  # number of vertices
        return len(self.vertices)

    def add_vertex(self, name: str, value: npt.NDArray = np.array([])):
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = Vertex(name, value)

    def add_edge(
        self, left_name: str, right_name: str, edge_name: str = None, cost: float = None
    ):
        if edge_name is None:
            edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists"
        self.edges[edge_name] = Edge(
            self.vertices[left_name], self.vertices[right_name], edge_name, cost
        )
        self.vertices[left_name].add_edge_out(edge_name)
        self.vertices[right_name].add_edge_in(edge_name)

    def add_cost_on_edge(self, edge_name: str, cost: float):
        self.edges[edge_name].set_cost(cost)

    def build_dual_optimization_program(self):
        raise Exception("Not Implemented")

    def set_start_target(self, start_name: str, target_name: str):
        self.start = start_name
        self.target = target_name

    def build_primal_optimization_program(self, convex_relaxation=True):
        assert self.start is not None
        assert self.target is not None
        assert self.start in self.vertices
        assert self.target in self.vertices

        self.primal_prog = MathematicalProgram()

        # for each edge, add decision variables: phi, y, z
        for e in self.edges.values():
            e.set_y(self.primal_prog.NewContinuousVariables(1, "y_" + e.name)[0])
            e.set_z(self.primal_prog.NewContinuousVariables(1, "z_" + e.name)[0])
            if convex_relaxation:
                e.set_phi(
                    self.primal_prog.NewContinuousVariables(1, "phi_" + e.name)[0]
                )
            else:
                e.set_phi(self.primal_prog.NewBinaryVariables(1, "phi_" + e.name)[0])

        # for each edge, add constraints
        for e in self.edges.values():
            # some tricks related to set inclusion
            order_box = Box(lb=np.array([1]), ub=np.array([self.n - 1]), state_dim=1)
            A1, b1 = order_box.get_perspective_hpolyhedron()

            order_box = Box(lb=np.array([2]), ub=np.array([self.n - 1]), state_dim=1)
            A2, b2 = order_box.get_perspective_hpolyhedron()

            if e.left.name == self.start:
                order_box_origin = Box(lb=np.array([0]), ub=np.array([0]), state_dim=1)
                oA, ob = order_box_origin.get_perspective_hpolyhedron()
                self.primal_prog.AddLinearConstraint(
                    le(oA @ np.array([e.y, e.phi]), ob)
                )
            elif e.left.name[0] == "s":
                self.primal_prog.AddLinearConstraint(
                    le(A1 @ np.array([e.y, e.phi]), b1)
                )
            else:
                self.primal_prog.AddLinearConstraint(
                    le(A2 @ np.array([e.y, e.phi]), b2)
                )

            if e.right.name == self.target:
                target_box = Box(
                    lb=np.array([self.n - 2]), ub=np.array([self.n - 2]), state_dim=1
                )
                tA, tb = target_box.get_perspective_hpolyhedron()
                self.primal_prog.AddLinearConstraint(
                    le(tA @ np.array([e.y, e.phi]), tb)
                )
                self.primal_prog.AddLinearConstraint(
                    le(A2 @ np.array([e.z, e.phi]), b2)
                )
            elif e.right.name[0] == "s":
                self.primal_prog.AddLinearConstraint(
                    le(A1 @ np.array([e.z, e.phi]), b1)
                )
            else:
                self.primal_prog.AddLinearConstraint(
                    le(A2 @ np.array([e.z, e.phi]), b2)
                )

            # order increase constraint
            self.primal_prog.AddLinearConstraint(e.y + e.phi == e.z)
            self.primal_prog.AddLinearConstraint(e.phi, 0.0, 1.0)

        # for each vertex, add constraints
        for v in self.vertices.values():
            if v.name != self.start:
                # add "flow in is 1" constraint
                flow_in = sum([self.edges[e].phi for e in v.edges_in])
                self.primal_prog.AddLinearConstraint(flow_in == 1)
            if v.name != self.target:
                # add flow out is 1 constraint
                flow_out = sum([self.edges[e].phi for e in v.edges_out])
                self.primal_prog.AddLinearConstraint(flow_out == 1)

            # sum of ys = sum of zs
            sum_of_y = sum([self.edges[e].y for e in v.edges_out])
            sum_of_z = sum([self.edges[e].z for e in v.edges_in])

            if v.name == self.start:
                self.primal_prog.AddLinearConstraint(sum_of_y == 0.0)
            elif v.name == self.target:
                self.primal_prog.AddLinearConstraint(sum_of_z == self.n - 1)
            else:
                self.primal_prog.AddLinearConstraint(sum_of_y == sum_of_z)

        # sum of left is sum of even, some of right is sum of odd
        left_vs = set()
        right_vs = set()
        for v in self.vertices.values():
            if v.name == "s0":
                left_vs.add(v.name)
            elif v.name == "t0":
                right_vs.add(v.name)
            elif v.name[0] == "t":
                left_vs.add(v.name)
            else:
                right_vs.add(v.name)
        left_pot_sum = (self.n / 2) * (self.n / 2 - 1)
        right_pot_sum = (self.n - 1) * self.n / 2 - left_pot_sum
        self.primal_prog.AddLinearConstraint(
            sum([e.z for e in self.edges.values() if e.right.name in left_vs])
            == left_pot_sum
        )
        self.primal_prog.AddLinearConstraint(
            sum([e.z for e in self.edges.values() if e.right.name in right_vs])
            == right_pot_sum
        )

        # total sum is given; don't need it if i already sum up left/right individually
        # self.primal_prog.AddLinearConstraint( sum( [e.z for e in self.edges.values()]) == (self.n-1)*self.n/2 )
        # self.primal_prog.AddLinearConstraint( sum( [e.y for e in self.edges.values()]) == (self.n-2)*(self.n-1)/2 )

        # add cost
        self.primal_prog.AddLinearCost(
            sum([e.phi * e.cost for e in self.edges.values()])
        )

    def solve_primal(self, convex_relaxation=True, verbose=False):
        # build the program
        x = timeit()
        self.build_primal_optimization_program(convex_relaxation)
        x.dt("Building the program")

        # solve
        self.primal_solution = Solve(self.primal_prog)
        x.dt("Solving the program")

        if self.primal_solution.is_success():
            YAY("Optimal primal cost is %.5f" % self.primal_solution.get_optimal_cost())
        else:
            ERROR("PRIMAL SOLVE FAILED!")
            ERROR(
                "Optimal primal cost is %.5f" % self.primal_solution.get_optimal_cost()
            )
            # ERROR(self.primal_solution.get_solver_details())
            return

        flows = [self.primal_solution.GetSolution(e.phi) for e in self.edges.values()]
        not_tight = np.any(
            np.logical_and(0.01 < np.array(flows), np.array(flows) < 0.99)
        )
        if not_tight:
            WARN("CONVEX RELAXATION NOT TIGHT")
        else:
            YAY("CONVEX RELAXATION IS TIGHT")

        if verbose:
            self.verbose_solution()

    def verbose_solution(self):
        flow_vars = [
            (e.name, self.primal_solution.GetSolution(e.phi))
            for e in self.edges.values()
        ]
        for (name, flow) in flow_vars:
            if flow > 0.01:
                print(name, flow)

        pots = []
        for v in self.vertices.values():
            sum_of_y = [
                self.primal_solution.GetSolution(self.edges[e].y) for e in v.edges_out
            ]
            sum_of_z = [
                self.primal_solution.GetSolution(self.edges[e].z) for e in v.edges_in
            ]
            print(v.name, sum_of_y, sum_of_z)
            sum_of_y = sum(
                [self.primal_solution.GetSolution(self.edges[e].y) for e in v.edges_out]
            )
            sum_of_z = sum(
                [self.primal_solution.GetSolution(self.edges[e].z) for e in v.edges_in]
            )
            pots.append((v.name, sum_of_z))

        # pots = [name for (name, _) in sorted(pots, key = lambda x: x[1])]
        pots = [x for x in sorted(pots, key=lambda x: x[1])]
        print(pots)

        left_vs = set()
        right_vs = set()
        for v in self.vertices.values():
            if v.name == "s0":
                left_vs.add(v.name)
            elif v.name == "t0":
                right_vs.add(v.name)
            elif v.name[0] == "t":
                left_vs.add(v.name)
            else:
                right_vs.add(v.name)
        print(left_vs)
        print(right_vs)
        print(
            sum(
                [
                    self.primal_solution.GetSolution(e.z)
                    for e in self.edges.values()
                    if e.right.name in left_vs
                ]
            )
        )
        print(
            sum(
                [
                    self.primal_solution.GetSolution(e.z)
                    for e in self.edges.values()
                    if e.right.name in right_vs
                ]
            )
        )
        left_pot_sum = (self.n / 2) * (self.n / 2 - 1)
        right_pot_sum = (self.n - 1) * self.n / 2 - left_pot_sum
        print(left_pot_sum, right_pot_sum)

        print(sum([self.primal_solution.GetSolution(e.y) for e in self.edges.values()]))
        print(sum([self.primal_solution.GetSolution(e.z) for e in self.edges.values()]))


def build_block_moving_gcs_tsp(
    start: npt.NDArray, target: npt.NDArray, block_dim: int, num_blocks: int
) -> TSPasGCS:
    bd = block_dim
    num_objects = num_blocks + 1
    # check lengths
    assert len(start) == block_dim * num_objects
    assert len(target) == block_dim * num_objects
    # naming
    def s(i):
        return "s" + str(i)

    def t(i):
        return "t" + str(i)

    def e(i, j):
        return i + "_" + j

    gcs = TSPasGCS()

    # add all vertices
    for i in range(num_objects):
        start_i = start[i * bd : i * bd + bd]
        target_i = target[i * bd : i * bd + bd]
        # pre-processing: if start_i = target_i, there is no need to move that object
        # must do same check over edges
        # if not (i > 0 and np.allclose(start_i, target_i)):
        gcs.add_vertex(s(i), start_i)
        gcs.add_vertex(t(i), target_i)

    # add all edges
    # add edge to from initial arm location to final arm location
    gcs.add_edge(s(0), t(0), e(s(0), t(0)))
    for i in range(1, num_objects):
        # start is connected to any object start locations
        gcs.add_edge(s(0), s(i), e(s(0), s(i)))
        # after ungrasping any object, we can move to arm target location
        gcs.add_edge(t(i), t(0), e(t(i), t(0)))
        # once we pick up an object, we must move it to the goal
        gcs.add_edge(s(i), t(i), e(s(i), t(i)))
        # after ungrasping an object, we can go and pick up any other object
        for j in range(1, num_objects):
            if i != j:
                gcs.add_edge(t(i), s(j), e(t(i), s(j)))

    # for each edge, add the cost on the edge
    for e in gcs.edges.values():
        e.set_cost(np.linalg.norm(e.left.value - e.right.value))

    # set start and target
    gcs.set_start_target("s0", "t0")
    return gcs
