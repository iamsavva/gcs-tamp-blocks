from gcs_for_blocks.motion_planning_obstacles_on_off import MotionPlanning
from gcs_for_blocks.axis_aligned_set_tesselation_2d import (
    Box,
    AlignedSet,
    plot_list_of_aligned_sets,
)
import numpy as np

mp = MotionPlanning()

bounding_box = AlignedSet(b=0, a=12, l=0, r=12)
block_width = 1
start = [(1, 1), (3, 5), (7, 4)]
target = [(5, 11), (9, 7), (5, 8)]

visitations = [1, 1, 1]
moving_block_index = 1
convex_relaxation = False

mp.set_bounding_box_aligned_set(bounding_box)
mp.build_the_graph(start, target, visitations, moving_block_index)
loc_path = mp.build_the_program(convex_relaxation)

plot_list_of_aligned_sets(mp.convex_sets, bounding_box, visitations, moving_block_index, loc_path)
