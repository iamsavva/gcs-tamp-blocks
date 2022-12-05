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
from draw_2d import Draw2DSolution

bounding_box = AlignedSet(b=0, a=5, l=0, r=10)
ub = np.array([bounding_box.r+1, bounding_box.a])
block_width = 1

start = [(1, 4), (1, 1), (1,3), (3,3), (3, 1)]
target = [(1, 4), (9, 1), (7,1), (7,3), (9, 3)]

convex_relaxation = False

x = timeit()
prog = BlockMovingObstacleAvoidance(start, target, bounding_box, block_width, convex_relaxation)
x.dt("Building the program")
prog.solve()
poses, modes = prog.get_drawing_stuff()
# [print(p,m) for (p,m) in zip(poses,modes)]

# visitations = [0,0,0,0]
# plot_list_of_aligned_sets(prog.convex_sets, bounding_box, visitations,0)

# print(poses)

tpose = prog.target_pos.copy()
tpose.resize(tpose.size)
drawer = Draw2DSolution(prog.num_blocks+1, ub, modes, poses, tpose, fast=True, no_arm = False)  # type: ignore
drawer.draw_solution()