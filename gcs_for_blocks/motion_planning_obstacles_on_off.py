import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import le, eq

from .axis_aligned_set_tesselation_2d import (
    AlignedSet,
    axis_aligned_tesselation,
    locations_to_aligned_sets,
    Box,
)
from .tsp_solver import Vertex, Edge, TSPasGCS


class MotionPlanning(TSPasGCS):
    def __init__(self):
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.vertices = dict()  # type:  T.Dict[str, Vertex]
        self.start = None  # str
        self.target = None  # str

        self.prog = None  # type: MathematicalProgram
        self.solution = None

        self.bounding_box = None
        self.num_blocks = None
        self.convex_set = None
        self.moving_block_index = None

    def set_bounding_box(self, l, r, a, b):
        self.bounding_box = AlignedSet(l=l, r=r, a=a, b=b)

    def build_the_graph(
        self,
        start_pos: T.List[T.Tuple[float,float]],
        target_pos: T.List[T.Tuple[float,float]],, 
        visitations: npt.NDArray,
        moving_block_index: int,
        block_width: float = 1.0,
    ):
        start_block_pos = start_pos[1:]
        target_block_pos = target_pos[1:]
        num_blocks = len(start_block_pos)
        self.start_block_pos = [np.array(x) for x in start_block_pos]
        self.target_block_pos = [np.array(x) for x in target_block_pos]

        self.start = "s" + str(moving_block_index) + "_tsp"
        self.target = "t" + str(moving_block_index) + "_tsp"
        self.num_blocks = len(start_block_pos)
        self.visitations = visitations
        self.moving_block_index = moving_block_index
        
        assert visitations[moving_block_index - 1] == 1
        assert len(target_block_pos) == num_blocks
        assert len(visitations) == num_blocks
        assert self.bounding_box is not None

        # get obstacles
        obstacles = locations_to_aligned_sets(start_block_pos, target_block_pos, block_width)

        # make a tesselation
        self.convex_sets = axis_aligned_tesselation(self.bounding_box.copy(), obstacles)

        ############################
        # add all vertices
        ############################

        # add start vertex
        self.add_vertex(self.start)
        # add target vertex
        self.add_vertex(self.target)
        # add set vertices
        for con_set in self.convex_sets:
            self.add_vertex(con_set.name)

        ############################
        # add all edges
        ############################
        # add edge from si to appropriate set, from ti to appropriate set
        self.add_edge(self.start, "s" + str(moving_block_index))
        self.add_edge("t" + str(moving_block_index), self.target)
        # add all edges
        for set1 in self.convex_sets:
            for set2 in self.convex_sets:
                # no repeats
                if set1 != set2 and set1.share_edge(set2):
                    # add edge between set1 and set2
                    self.add_edge(set1.name, set2.name)


    def build_the_program(self, convex_relaxation=True):
        self.prog = MathematicalProgram()

        self.add_variables()

        
    def add_variables(self, convex_relaxation=True):
        ####################################
        # add variables to start and target vertices
        # associated vaiables are visitations, n x 1, each 0 or 1
        self.vertices[self.start].set_v(self.prog.NewContinuousVariables(self.num_blocks))
        self.vertices[self.target].set_v(self.prog.NewContinuousVariables(self.num_blocks))
        # no other vertex has variables associated with it

        ###################################
        # add edge variables
        for e in self.edges:
            # add flow variable
            if convex_relaxation:
                e.set_phi(self.primal_prog.NewContinuousVariables(1, "phi_" + e.name)[0])
            else:
                e.set_phi(self.primal_prog.NewBinaryVariables(1, "phi_" + e.name)[0])

            # if the edge is not from start / to target 
            if e.left.name != self.start and e.right.name != self.target:
                # set left and right position variables, 
                e.set_y(self.primal_prog.NewContinuousVariables(2, "y_" + e.name))
                e.set_z(self.primal_prog.NewContinuousVariables(2, "z_" + e.name))


    def add_constraints(self):
        ###################################
        # visitiations box constraints
        visitation_box = Box(lb=np.zeros(self.num_blocks), ub=np.ones(self.num_blocks), state_dim=self.num_blocks)
        vA, vb = visitation_box.get_hpolyhedron()

        ###################################
        for v in self.vertices:
            # sum of ys = sum of zs
            sum_of_y = sum([self.edges[e].y for e in v.edges_out])
            sum_of_z = sum([self.edges[e].z for e in v.edges_in])

            # it's a start node
            if v.name == self.start:
                # add visitations box constraint
                self.prog.AddLinearConstraint(le(vA @ v.v, vb))
                # add visitation equality constraint
                self.prog.AddLinearConstraint(eq(v.v, self.visitations))
                
                # flow in is 1

            # it's a target node
            elif v.name == self.target:
                # add visitations box constraint
                self.prog.AddLinearConstraint(le(vA @ v.v, vb))
                # add visitation equality constraint to visitations of start
                self.prog.AddLinearConstraint(eq(v.v, self.vertices[self.start].v))
            # it's a start-set node
            elif v.name == "s"+str(self.moving_block_index):
                # sum pos out is the start-pos
                block_start_pos = self.start_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint( eq( sum_of_y, block_start_pos ) )
            # it's a target-set node
            elif v.name == "t"+str(self.moving_block_index):
                # sum pos out is the target-pos 
                block_target_pos = self.target_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint( eq( sum_of_z, block_target_pos ) )
            else:


                

        # add constraints on start / target set vertices
        start_set_name = "s"+str(self.moving_block_index)
        target_set_name = "t"+str(self.moving_block_index)
        start_set_vertex = self.vertices[start_set_name]
        target_set_vertex = self.vertices[target_set_name]




