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

    def set_bounding_box(self, l, r, a, b):
        self.bounding_box = AlignedSet(l=l, r=r, a=a, b=b)

    def build_program(
        self,
        start_pos,
        target_pos,
        visitations,
        moving_block_index: int,
        block_width: float = 1.0,
    ):
        self.start = "s" + str(moving_block_index) + "_tsp"
        self.target = "t" + str(moving_block_index) + "_tsp"

        start_block_pos = start_pos[1:]
        target_block_pos = target_pos[1:]
        num_blocks = len(start_block_pos)
        assert visitations[moving_block_index - 1] == 1
        assert len(target_block_pos) == num_blocks
        assert len(visitations) == num_blocks

        # get obstacles
        obstacles = locations_to_aligned_sets(start_block_pos, target_block_pos, block_width)

        # make a tesselation
        convex_sets = axis_aligned_tesselation(self.bounding_box.copy(), obstacles)

        ############################
        # add all vertices
        ############################
        self.prog = MathematicalProgram()

        # add start and target vertices
        visitation_box = Box(lb=np.zeros(num_blocks), ub=np.ones(num_blocks), state_dim=num_blocks)
        vA, vb = visitation_box.get_hpolyhedron()

        # add start vertex
        self.add_vertex(self.start)
        # add visitiation variable
        self.vertices[self.start].set_v(self.prog.NewContinuousVariables(num_blocks))
        # add visitations box constraint
        self.prog.AddLinearConstraint(le(vA @ self.vertices[self.start].v, vb))
        # add visitation equality constraint
        self.prog.AddLinearConstraint(eq(self.vertices[self.start].v, visitations))

        # add target vertex
        self.add_vertex(self.target)
        # add visitiation variable
        self.vertices[self.target].set_v(self.prog.NewContinuousVariables(num_blocks))
        # add visitations box constraint
        self.prog.AddLinearConstraint(le(vA @ self.vertices[self.target].v, vb))
        # add visitation equality constraint to start vertex
        self.prog.AddLinearConstraint(eq(self.vertices[self.target].v, self.vertices[self.start].v))

        # associated variables are visitations, nx1, each binary

        # for each set, add the set as a vertex
        # associated variables are x, y location
        for con_set in convex_sets:
            # add set vertex
            self.add_vertex(con_set.name)
            # this vertex has no variable of its own, so this is it

        ############################
        # add all edges
        ############################
        # add edge from si to appropriate set, from ti to appropriate

        for set1 in convex_sets:
            for set2 in convex_sets:
                # no repeats
                if set1 != set2:
                    # if the two intersect
                    if set1.share_edge(set2):
                        raise Exception("unimplemented")
                        # add edge between set1 and set2

                        # add all variables

                        # add all edge constraints

                        # add edge cost

        # for each vertex
        # add all vertex constraints
