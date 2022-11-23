import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve


class Vertex:
    def __init__(self, value: npt.NDArray, name: str):
        self.value = value
        self.name = name
        self.left_nbhs = []
        self.right_nbhs = []
        self.var = None

    def add_left_neighbor(self, nbh: str):
        assert nbh not in self.left_nbhs
        self.left_nbhs.append(nbh)

    def add_right_neighbor(self, nbh: str):
        assert nbh not in self.right_nbhs
        self.right_nbhs.append(nbh)

    def set_var(self, var):
        # variable is either a float or a ContinuousVariable
        assert self.var is None, "Var for " + self.name + " is already set"
        self.var = var


class Edge:
    def __init__(
        self, left_vertex: Vertex, right_vertex: Vertex, name: str, cost: float = None
    ):
        self.left = left_vertex
        self.right = right_vertex
        self.name = name
        self.cost = cost
        self.var = None

    def set_cost(self, cost: float):
        assert self.cost is None, "Cost for " + self.name + " is already set"
        self.cost = cost

    def set_var(self, var):
        assert self.var is None, "Var for " + self.name + " is already set"
        self.var = var


class DiscreteNetworkFlowGraph:
    def __init__(self):
        self.edges = dict()  # T.Dict[str, Edge]
        self.vertices = dict()  # T.Dict[str, Vertex]
        self.prog = None
        self.result = None

    @property
    def n(self):
        return len(self.vertices)

    def add_vertex(self, value: npt.NDArray, name: str):
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = Vertex(value, name)

    def add_edge(self, left_name: str, right_name: str, edge_name: str, cost=None):
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists"
        self.edges[edge_name] = Edge(
            self.vertices[left_name], self.vertices[right_name], edge_name, cost
        )
        self.vertices[left_name].add_right_neighbor(edge_name)
        self.vertices[right_name].add_left_neighbor(edge_name)

    def add_cost_on_edge(self, edge_name: str, cost: float):
        self.edges[edge_name].set_cost(cost)

    def build_from_start_and_target(
        self, start: npt.NDArray, target: npt.NDArray, block_dim: int, num_blocks: int
    ):
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

        # add all vertices
        for i in range(num_objects):
            start_i = start[i * bd : i * bd + bd]
            target_i = target[i * bd : i * bd + bd]
            # pre-processing: if start_i = target_i, there is no need to move that object
            if not (i > 0 and np.allclose(start_i, target_i)):
                self.add_vertex(start_i, s(i))
                self.add_vertex(target_i, t(i))

        # add all edges

        # add edge to from initial arm location to final arm location
        self.add_edge(s(0), t(0), e(s(0), t(0)))
        for i in range(1, num_objects):
            # start is connected to any object start locations
            self.add_edge(s(0), s(i), e(s(0), s(i)))
            # after ungrasping any object, we can move to arm target location
            self.add_edge(t(i), t(0), e(t(i), t(0)))
            # once we pick up an object, we must move it to the goal
            self.add_edge(s(i), t(i), e(s(i), t(i)))
            # after ungrasping an object, we can go and pick up any other object
            for j in range(num_objects):
                if i != j:
                    self.add_edge(t(i), s(j), e(t(i), s(j)))

        # for each edge, add the cost on the edge
        for e in self.edges.values():
            e.set_cost(np.linalg.norm(e.left.value - e.right.value))

    def solve(self, convex_relaxation = True):
        start = "s0"
        target = "t0"
        # is this inefficient or not?
        # in practice, shouldn't I build an edge matrix?
        x = timeit()

        self.prog = MathematicalProgram()

        # add flow decision variables
        for e in self.edges.values():
            if convex_relaxation:
                e.set_var(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
                # add flow 0 to 1 constraint
                self.prog.AddBoundingBoxConstraint(0.0, 1.0, e.var)
            else:
                e.set_var(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])


        for v in self.vertices.values():
            # add vertex potential variables
            v.set_var(self.prog.NewContinuousVariables(1, "u_" + v.name)[0])
            if v.name != start:
                # add "flow in is 1" constraint
                flow_in = sum([self.edges[e].var for e in v.left_nbhs])
                self.prog.AddConstraint(flow_in == 1)
                # potential is between 1 and n-1
                self.prog.AddBoundingBoxConstraint(1.0, self.n - 1.0, v.var)
            else:
                # potential at start is 0
                self.prog.AddConstraint(v.var == 0)

            # add flow out is 1 constraint
            if v.name != target:
                flow_out = sum([self.edges[e].var for e in v.right_nbhs])
                self.prog.AddConstraint(flow_out == 1)

        # for each edge, generate a flow variable
        for e in self.edges.values():
            # increasing potential over edge flow
            pot_diff = e.right.var + self.n - 2 - e.left.var - (self.n - 1) * e.var
            self.prog.AddConstraint(pot_diff >= 0)

        # add cost
        self.prog.AddLinearCost(sum([e.var * e.cost for e in self.edges.values()]))
        x.dt("Building the program")

        # solve
        self.result = Solve(self.prog)
        x.dt("Solving the program")

        INFO(f"Is solved successfully: {self.result.is_success()}")
        print(f"optimal cost: {self.result.get_optimal_cost()}")

        flows = [(e.name, self.result.GetSolution(e.var)) for e in self.edges.values()]
        for name, flow in flows:
            if flow > 0:
                print(name, flow)

        potentials = [
            (v.name, self.result.GetSolution(v.var)) for v in self.vertices.values()
        ]
        # sort by potential
        potentials.sort(key=lambda y: y[1])
        for value, potential in potentials:
            print(value, potential)
