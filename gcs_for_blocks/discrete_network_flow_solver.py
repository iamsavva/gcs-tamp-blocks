import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve

# import graphviz


class Vertex:
    def __init__(self, value: npt.NDArray, name: str):
        self.value = value
        self.name = name
        self.left_nbhs = []
        self.right_nbhs = []
        self.var = None

        self.r = None
        self.s = None

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

    def set_duals(self, r, s):
        self.r = r
        self.s = s


class Edge:
    def __init__(self, left_vertex: Vertex, right_vertex: Vertex, name: str, cost: float = None):
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
        self.start = start
        self.target = target
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

    def build_dual_optimization_program(self):
        start = "s0"
        target = "t0"
        self.dual_prog = MathematicalProgram()

        for v in self.vertices.values():
            r = self.dual_prog.NewContinuousVariables(1, "r_" + v.name)[0]
            s = self.dual_prog.NewContinuousVariables(1, "s_" + v.name)[0]
            self.dual_prog.AddBoundingBoxConstraint(-0, 1000, r)
            self.dual_prog.AddBoundingBoxConstraint(-1000, 0, s)
            v.set_duals(r, s)
            if v.name == start:
                self.dual_prog.AddConstraint(v.r == 0.0)
            if v.name == target:
                self.dual_prog.AddConstraint(v.s == 0.0)

        for e in self.edges.values():
            self.dual_prog.AddConstraint(e.cost + e.right.r + e.left.s >= 0)

        self.dual_prog.AddLinearCost(sum([(v.r + v.s) for v in self.vertices.values()]))
        # solve
        self.dual_result = Solve(self.dual_prog)
        # x.dt("Solving the program")

        if self.dual_result.is_success():
            YAY("Optimal cost is %.5f" % self.dual_result.get_optimal_cost())
        else:
            ERROR("SOLVE FAILED!")

        # get potentials
        r_pots = [
            (
                v.name,
                self.dual_result.GetSolution(v.r),
                self.dual_result.GetSolution(v.s),
            )
            for v in self.vertices.values()
        ]
        # sort by potential
        for value, r, s in r_pots:
            print(value, r, s)

    def build_primal_optimization_program(self, convex_relaxation=True, add_potentials=True):
        start = "s0"
        target = "t0"
        # is this inefficient or not?
        # in practice, shouldn't I build an edge matrix?

        self.prog = MathematicalProgram()

        # add flow decision variables
        for e in self.edges.values():
            if convex_relaxation:
                e.set_var(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
                # add flow 0 to 1 constraint
                self.prog.AddConstraint(e.var >= 0.0)
            else:
                e.set_var(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])

        for v in self.vertices.values():
            if v.name != start:
                # add "flow in is 1" constraint
                flow_in = sum([self.edges[e].var for e in v.left_nbhs])
                self.prog.AddConstraint(flow_in == 1)
            if v.name != target:
                # add flow out is 1 constraint
                flow_out = sum([self.edges[e].var for e in v.right_nbhs])
                self.prog.AddConstraint(flow_out == 1)

        if add_potentials:
            # add vertex potential variables
            for v in self.vertices.values():
                v.set_var(self.prog.NewContinuousVariables(1, "u_" + v.name)[0])
                if v.name != start:
                    # potential is between 1 and n-1
                    self.prog.AddBoundingBoxConstraint(1.0, self.n - 1.0, v.var)
                else:
                    # potential at start is 0
                    self.prog.AddConstraint(v.var == 0)

            # for each edge, increasing potential over edge flow
            for e in self.edges.values():
                pot_diff = e.right.var + self.n - 2 - e.left.var - (self.n - 1) * e.var
                self.prog.AddConstraint(pot_diff >= 0)

        # add cost
        self.prog.AddLinearCost(sum([e.var * e.cost for e in self.edges.values()]))

    def solve_primal(self, convex_relaxation=True):
        # build the program
        x = timeit()
        self.build_primal_optimization_program(convex_relaxation)
        x.dt("Building the program")

        # solve
        self.result = Solve(self.prog)
        x.dt("Solving the program")

        if self.result.is_success():
            YAY("Optimal cost is %.5f" % self.result.get_optimal_cost())
        else:
            ERROR("SOLVE FAILED!")

        # # get flow results
        # flows = [(e.name, self.result.GetSolution(e.var)) for e in self.edges.values()]
        # for name, flow in flows:
        #     if flow > 0:
        #         print(name, flow)

        # get potentials
        potentials = [(v.name, self.result.GetSolution(v.var)) for v in self.vertices.values()]
        # sort by potential
        potentials.sort(key=lambda y: y[1])
        for value, potential in potentials:
            print(value, potential)

    def draw(self):
        # get potentials
        potentials = [(v.name, self.result.GetSolution(v.var)) for v in self.vertices.values()]

        # # get flow results
        # flows = [(e.name, self.result.GetSolution(e.var)) for e in self.edges.values()]

        # f = graphviz.Digraph('', filename='fsm.gv')

        # f.attr(rankdir='LR', size='8,5')
        # f.attr('node', shape='doublecircle')
        # f.node('LR_0')
        # f.node('LR_3')
        # f.node('LR_4')
        # f.node('LR_8')

        # f.attr('node', shape='circle')
        # f.edge('LR_0', 'LR_2', label='SS(B)')
        # f.edge('LR_0', 'LR_1', label='SS(S)')
        # f.edge('LR_1', 'LR_3', label='S($end)')
        # f.edge('LR_2', 'LR_6', label='SS(b)')
        # f.edge('LR_2', 'LR_5', label='SS(a)')
        # f.edge('LR_2', 'LR_4', label='S(A)')
        # f.edge('LR_5', 'LR_7', label='S(b)')
        # f.edge('LR_5', 'LR_5', label='S(a)')
        # f.edge('LR_6', 'LR_6', label='S(b)')
        # f.edge('LR_6', 'LR_5', label='S(a)')
        # f.edge('LR_7', 'LR_8', label='S(b)')
        # f.edge('LR_7', 'LR_5', label='S(a)')
        # f.edge('LR_8', 'LR_6', label='S(b)')
        # f.edge('LR_8', 'LR_5', label='S(a)')

        # f.view()
