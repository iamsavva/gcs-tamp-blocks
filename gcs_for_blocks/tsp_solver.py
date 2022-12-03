import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import le, eq

# import graphviz


class Vertex:
    def __init__(self, value: npt.NDArray, name: str):
        self.value = value # effectively just the name
        self.name = name # name of the vertex
        self.edges_in = [] # str names of edges in 
        self.edges_out = [] # str names of edges out

    def add_edge_in(self, nbh: str):
        assert nbh not in self.edges_in
        self.edges_in.append(nbh)

    def add_edge_out(self, nbh: str):
        assert nbh not in self.edges_out
        self.edges_out.append(nbh)

class Edge:
    def __init__(
        self, left_vertex: Vertex, right_vertex: Vertex, name: str, cost: float = None
    ):
        self.left = left_vertex
        self.right = right_vertex
        self.name = name
        self.cost = cost

        # primal variables
        self.phi = None
        self.y = None
        self.z = None

    def set_cost(self, cost: float):
        assert self.cost is None, "Cost for " + self.name + " is already set"
        self.cost = cost

    def set_phi(self, flow):
        assert self.phi is None, "Flow for " + self.name + " is already set"
        self.phi = flow

    def set_y(self, y):
        assert self.y is None, "y for " + self.name + " is already set"
        self.y = y

    def set_z(self, z):
        assert self.z is None, "z for " + self.name + " is already set"
        self.z = z

class TSPasGCS:
    def __init__(self):
        self.edges = dict()  # T.Dict[str, Edge]
        self.vertices = dict()  # T.Dict[str, Vertex]
        self.start = None # str
        self.target = None # str
        self.primal_prog = None # MathematicalProgram
        self.primal_solution = None 

    @property
    def n(self): # number of vertices
        return len(self.vertices)

    def add_vertex(self, value: npt.NDArray, name: str):
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = Vertex(value, name)

    def add_edge(self, left_name: str, right_name: str, edge_name: str, cost=None):
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

    def set_start_target(self, start_name:str, target_name:str):
        self.start = start_name
        self.target = target_name
    
    def build_primal_optimization_program(self, convex_relaxation = True):
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
                e.set_phi(self.primal_prog.NewContinuousVariables(1, "phi_" + e.name)[0])
            else:
                e.set_phi(self.primal_prog.NewBinaryVariables(1, "phi_" + e.name)[0])
            
        # for each edge, add constraints
        for e in self.edges.values():
            # TODO: formulate this nicer through Ax < bphi
            A = np.array( [[-(self.n-1), 1], [-1, 0], [0, 1]] )
            b = np.array( [0, 0, self.n-1] )
            # flow and left variable belong to an order increase cone
            self.primal_prog.AddLinearConstraint( le(A @ np.array([e.phi, e.y]), b) )
            # flow and right variable belong to an order increase cone
            self.primal_prog.AddLinearConstraint( le(A @ np.array([e.phi, e.z]), b) )
            # order increase constraint
            self.primal_prog.AddLinearConstraint( e.y + e.phi == e.z )

        # for each vertex, add constraints
        for v in self.vertices.values():
            if v.name != self.start:
                # add "flow in is 1" constraint
                flow_in = sum([self.edges[e].phi for e in v.edges_in])
                self.primal_prog.AddLinearConstraint(flow_in == 1)
            if v.name != self.target:
                # add flow out is 1 constraint
                flow_out = sum([self.edges[e].phi for e in v.edges_out])
            
            # sum of ys = sum of zs
            sum_of_y = sum([self.edges[e].y for e in v.edges_out])
            sum_of_z = sum([self.edges[e].z for e in v.edges_in])

            if v.name == self.start:
                self.primal_prog.AddLinearConstraint(sum_of_y == 0.0)
            elif v.name == self.target:
                self.primal_prog.AddLinearConstraint(sum_of_z == self.n-1)
            else:
                self.primal_prog.AddLinearConstraint(sum_of_y == sum_of_z)

        # add cost
        self.primal_prog.AddLinearCost(sum([e.phi * e.cost for e in self.edges.values()]))
        
    def solve_primal(self, convex_relaxation = True):
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
            ERROR("Optimal primal cost is %.5f" % self.primal_solution.get_optimal_cost())
            # ERROR(self.primal_solution.get_solver_details())
            return

        flows = [self.primal_solution.GetSolution(e.phi) for e in self.edges.values()]
        not_tight = np.any(
            np.logical_and(0.01 < np.array(flows), np.array(flows) < 0.99)
        )
        if not_tight:
            WARN("SOLUTION NOT TIGHT")
        else:
            YAY("SOLUTION IS TIGHT")

def build_block_moving_gcs_tsp(start: npt.NDArray, target: npt.NDArray, block_dim: int, num_blocks: int) -> TSPasGCS:
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
        gcs.add_vertex(start_i, s(i))
        gcs.add_vertex(target_i, t(i))

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