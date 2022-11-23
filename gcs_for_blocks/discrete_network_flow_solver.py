import typing as T

import numpy as np
import numpy.typing as npt


# from PIL import Image as PIL_Image

import pydrake.geometry.optimization as opt  # pylint: disable=import-error
from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    GraphOfConvexSets,
    HPolyhedron,
    ConvexSet,
)

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve

class Vertex:
    def __init__(self, value: npt.NDArray, name:str):
        self.value = value
        self.name = name

        self.left_nbhs = []
        self.right_nbhs = []

        self.var = None

    def add_left_neighbor(self, nbh:str):
        assert nbh not in self.left_nbhs
        self.left_nbhs.append(nbh)

    def add_right_neighbor(self, nbh:str):
        assert nbh not in self.right_nbhs
        self.right_nbhs.append(nbh)

    def set_var(self, var):
        assert self.var is None, "Var for " + self.name + " is already set"
        self.var = var


class Edge:
    def __init__(self, left_vertex:Vertex, right_vertex:Vertex, name:str, cost:float = None):
        self.left = left_vertex
        self.right = right_vertex
        self.name = name
        self.cost = cost

        self.var = None 

    def set_cost(self, cost:float):
        assert self.cost is None, "Cost for " + self.name + " is already set"
        self.cost = cost

    def set_var(self, var):
        assert self.var is None, "Var for " + self.name + " is already set"
        self.var = var

class DiscreteNetworkFlowGraph:
    def __init__(self):
        self.edges = dict() # T.Dict[str, Edge]
        self.vertices = dict() # T.Dict[str, Vertex]
        self.prog = None
        self.result = None

    @property
    def n(self):
        return len(self.vertices)

    def add_vertex(self, value:npt.NDArray, name:str):
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        self.vertices[name] = Vertex(value, name)

    def add_edge(self, left_name:str, right_name:str, edge_name:str, cost=None):
        assert edge_name not in self.edges, "Edge with name " + edge_name + " already exists"
        self.edges[edge_name] = Edge(self.vertices[left_name], self.vertices[right_name], edge_name, cost)
        self.vertices[left_name].add_right_neighbor(edge_name)
        self.vertices[right_name].add_left_neighbor(edge_name)

    def add_cost_on_edge(self, edge_name:str, cost:float):
        self.edges[edge_name].set_cost(cost)

    def build_from_start_and_target(self, start, target, block_dim, num_blocks):
        assert len(start) == block_dim * (num_blocks+1)
        assert len(target) == block_dim * (num_blocks+1)
        bd = block_dim
        def s(i):
            return "s" +str(i)
        def t(i):
            return "t" +str(i)
        def e(i,j):
            return i+"_"+j

        # add all vertices
        for i in range(num_blocks+1):
            self.add_vertex(start[i*bd:i*bd+bd], s(i))
            self.add_vertex(target[i*bd:i*bd+bd], t(i))

        # TODO: add pre-processing: if start_i = target_i, don't add 
               
        self.add_edge(s(0), t(0), e(s(0),t(0)))
        for i in range(1, num_blocks+1):
            # start is connected to everything on the right
            self.add_edge(s(0), s(i), e(s(0),s(i)))
            # from s only to t
            self.add_edge(s(i), t(i), e(s(i),t(i)))
            # i is connect to target end
            self.add_edge(t(i), t(0), e( t(i),t(0)))

            # i is connected to anything except for itself
            for j in range(num_blocks+1):
                if i != j:
                    self.add_edge(t(i), s(j), e( t(i),s(j)))


        for e in self.edges.values():
            e.set_cost( np.linalg.norm(e.left.value-e.right.value) )
             
        self.solve("s0", "t0")


        

    def solve(self, start:str, target:str):
        # the inefficient way
        # in practice you should build an edge matrix and a vertex matrix

        x = timeit()

        self.prog = MathematicalProgram()

        for e in self.edges.values():
            # TODO: add option to generate discrete constraint
            e.set_var(self.prog.NewContinuousVariables(1, "phi_" + e.name)) 
            # add flow 0 to 1 constraint 
            self.prog.AddBoundingBoxConstraint(0.0, 1.0, e.var)

        for v in self.vertices.values():
            # add order variable
            
            if v.name != start:
                v.set_var(self.prog.NewContinuousVariables(1, "u_" + v.name))
                # add flow in is 1 constraint
                self.prog.AddConstraint(sum([self.edges[e].var for e in v.left_nbhs])[0] == 1)
                # order is between 1 and n-1
                self.prog.AddBoundingBoxConstraint(1.0, self.n-1.0, v.var)
            else:
                # order of start vertex is 0
                v.set_var(0.0)

            # add flow out is 1 constraint
            if v.name != target:
                self.prog.AddConstraint(sum([self.edges[e].var for e in v.right_nbhs])[0] == 1)

        # for each edge, generate a flow variable
        for e in self.edges.values():
            # increasing order constraints
            expr = e.right.var + self.n-2  - e.left.var - (self.n-1)*e.var
            self.prog.AddConstraint(expr[0] >=  0 )

        # add cost
        self.prog.AddLinearCost( sum([e.var[0] * e.cost for e in self.edges.values()]) )

        x.dt("Building the program")
        self.result = Solve(self.prog)
        x.dt("Solving the program")

        INFO(f"Is solved successfully: {self.result.is_success()}")
        # INFO(f"x optimal value: {self.result.GetSolution(x)}")
        print(f"optimal cost: {self.result.get_optimal_cost()}")


        flows = [(e.name, self.result.GetSolution(e.var)) for e in self.edges.values()]
        for name, flow in flows:
            if flow > 0.9:
                print(name, flow)

        pots = [(v.value, self.result.GetSolution(v.var)[0]) for v in self.vertices.values() if v.name != "s0"]
        pots += [("s0", 0)]
        pots.sort(key = lambda y: y[1])
        for name, val in pots:
            print(name, val)

