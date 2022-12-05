from gcs_for_blocks.motion_planning_obstacles_on_off import MotionPlanning
from gcs_for_blocks.axis_aligned_set_tesselation_2d import (
    Box,
    AlignedSet,
    plot_list_of_aligned_sets,
    locations_to_aligned_sets,
    axis_aligned_tesselation,
)
import numpy as np

from pydrake.solvers import MathematicalProgram, Solve
from pydrake.math import le, eq
from gcs_for_blocks.tsp_solver import Vertex, Edge
from gcs_for_blocks.util import timeit, INFO, WARN, ERROR, YAY
from gcs_for_blocks.tsp_obstacle_avoidance import BlockMovingObstacleAvoidance

bounding_box = AlignedSet(b=0, a=12, l=0, r=12)
block_width = 1

start = [(0, 0), (1, 1), (3, 5), (7, 4)]
target = [(0, 0), (5, 11), (9, 7), (5, 8)]
convex_relaxation = False

prog = BlockMovingObstacleAvoidance(start, target, bounding_box, block_width, convex_relaxation)
prog.solve()