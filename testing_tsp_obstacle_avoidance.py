import numpy as np

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgram,
    Solve,
) 
from pydrake.math import le, eq  # pylint: disable=import-error, no-name-in-module
from gcs_for_blocks.tsp_solver import Vertex, Edge
from gcs_for_blocks.util import timeit, INFO, WARN, ERROR, YAY
from gcs_for_blocks.tsp_obstacle_avoidance import BlockMovingObstacleAvoidance
from gcs_for_blocks.motion_planning_obstacles_on_off import MotionPlanning
from gcs_for_blocks.axis_aligned_set_tesselation_2d import (
    Box,
    AlignedSet,
    plot_list_of_aligned_sets,
    locations_to_aligned_sets,
    axis_aligned_tesselation,
)
from draw_2d import Draw2DSolution

#############################################################################

def solve_the_program(bounding_box, start, target, block_width, convex_relaxation=False, fast=True):
    set_tol = 0.00001
    share_edge_tol = set_tol/50
    block_width_minus_tol = block_width - set_tol
    half_block_width = block_width/2
    half_block_width_minus_tol = half_block_width - set_tol

    bounding_box.offset_in(half_block_width_minus_tol)


    x = timeit()
    prog = BlockMovingObstacleAvoidance(
        start_pos = start, target_pos = target, bounding_box = bounding_box, block_width = block_width_minus_tol, convex_relaxation = convex_relaxation, share_edge_tol=share_edge_tol
    )
    x.dt("Building the program")
    prog.solve()
    positions, modes = prog.get_trajectory_for_drawing()

    bounding_box.offset_in(-half_block_width_minus_tol)
    target_position = prog.target_pos.copy()
    target_position.resize(target_position.size)
    drawer = Draw2DSolution(
        prog.num_blocks + 1, np.array([bounding_box.r, bounding_box.a]), modes, positions, target_position, fast=fast, no_arm=False, no_padding=True
    )
    drawer.draw_solution()

#############################################################################

block_width = 1
bounding_box = AlignedSet(b=0, a=6, l=0, r=7)
start = [
    (1 - 0.5, 5 - 0.5),
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

bounding_box = AlignedSet(b=0, a=3, l=0, r=5)
start =  [(2.5, 1), (1.5, 0.5), (0.5, 1.5), (2-0.5, 2-0.5), (2-0.5, 1-0.5)]
target = [(2.5, 1), (4.5, 0.5), (4.5, 1.5), (4-0.5, 2-0.5), (4-0.5, 1-0.5)]

bounding_box = AlignedSet(b=0, a=2, l=0, r=5)

start =  [(3-0.5, 1.5-0.5), (1-0.5, 1-0.5), (1-0.5, 2-0.5), (2-0.5, 2-0.5), (2-0.5, 1-0.5)]
target = [(3-0.5, 1.5-0.5), (5-0.5, 1-0.5), (5-0.5, 2-0.5), (4-0.5, 2-0.5), (4-0.5, 1-0.5)] 

convex_relaxation = False

solve_the_program(bounding_box=bounding_box, start=start, target=target, block_width=block_width, convex_relaxation=convex_relaxation, fast=fast)