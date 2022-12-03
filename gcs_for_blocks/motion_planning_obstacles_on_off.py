import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import le, eq

from .axis_aligned_set_tesselation_2d import AlignedSet, axis_aligned_tesselation, locations_to_aligned_sets, Box


class MotionPlanning:

    def __init__(self):
        self.edges = dict()  # T.Dict[str, Edge]
        self.vertices = dict()  # T.Dict[str, Vertex]
        self.start = None # str
        self.target = None # str
        self.primal_prog = None # MathematicalProgram
        self.primal_solution = None 
        self.bounding_box = None

    def set_bounding_box(self, l, r, a, b):
        self.bounding_box = AlignedSet(l=l, r=r, a=a, b=b)

    def build_program(self, start_pos, target_pos, visitations, moving_block_index:int, block_width:float = 1.0):
        start_block_pos = start_pos[1:]
        target_block_pos = target_pos[1:]
        num_blocks = len(start_block_pos)
        assert visitations[moving_block_index-1] == 1
        assert len(target_block_pos) == num_blocks
        assert len(visitations) == num_blocks
        # get obstacles
        obstacles = locations_to_aligned_sets(start_block_pos, target_block_pos, block_width)

        # make a tesselation
        convex_sets = axis_aligned_tesselation(self.bounding_box.copy(), obstacles)

        ############################
        # add all vertices
        ############################

        # add start and target vertices
        visitation_convex_set = Box(lb=np.zeros(num_blocks), ub=np.ones(num_blocks), state_dim=num_blocks)
        # associated variables are visitations, nx1, each binary

        # for each set, add the set as a vertex
        # associated variables are x, y location
    
        # add all edges
        for set1 in convex_sets:
            for set2 in convex_sets:
                # no repeats
                if set1 != set2:
                    # if the two intersect
                    if set1.share_edge(set2):
                        # add edge between set1 and set2

                        # add all variables

                        # add all edge constraints

                        # add edge cost

        # for each vertex
        # add all vertex constraints






        