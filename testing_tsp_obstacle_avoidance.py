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

bounding_box = AlignedSet(b=0, a=6, l=0, r=7)
start = [
    (1 - 0.5, 4 - 0.5),
    (1 - 0.5, 1 - 0.5),
    (1 - 0.5, 3 - 0.5),
    (3 - 0.5, 3 - 0.5),
    (3 - 0.5, 1 - 0.5),
    (1 - 0.5, 5 - 0.5),
    (3 - 0.5, 5 - 0.5),
]
target = [
    (1 - 0.5, 6 - 0.5),
    (7 - 0.5, 1 - 0.5),
    (5 - 0.5, 1 - 0.5),
    (5 - 0.5, 3 - 0.5),
    (5 - 0.5, 5 - 0.5),
    (7 - 0.5, 5 - 0.5),
    (7 - 0.5, 3 - 0.5),
]
fast = True

# bounding_box = AlignedSet(b=0, a=3, l=0, r=7)
# start =  [(4-0.5, 2-0.5), (1-0.5, 1-0.5), (1-0.5, 3-0.5), (3-0.5, 3-0.5), (3-0.5, 1-0.5)]
# target = [(4-0.5, 2-0.5), (7-0.5, 1-0.5), (7-0.5, 3-0.5), (5-0.5, 3-0.5), (5-0.5, 1-0.5)]
# fast = False

ub = np.array([bounding_box.r, bounding_box.a])
block_width = 1

convex_relaxation = False

x = timeit()
# TODO: a somewhat bad fix
prog = BlockMovingObstacleAvoidance(
    start, target, bounding_box, block_width - 0.00001, convex_relaxation
)
x.dt("Building the program")
prog.solve()
poses, modes = prog.get_drawing_stuff()
# print(modes)
# [print(p,m) for (p,m) in zip(poses,modes)]

# visitations = [0,0,0,0]
# plot_list_of_aligned_sets(prog.convex_set_tesselation, bounding_box, visitations,0)

# print(poses)

tpose = prog.target_pos.copy()
tpose.resize(tpose.size)
drawer = Draw2DSolution(prog.num_blocks + 1, ub, modes, poses, tpose, fast=fast, no_arm=False, no_padding=True)  # type: ignore
drawer.draw_solution()
