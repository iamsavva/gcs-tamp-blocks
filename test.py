from pydrake.geometry.optimization import Point
import numpy as np

from gcs import GCSforBlocks
from util import INFO


def make_simple_transparent_gcs_test(
    block_dim: int,
    num_blocks: int,
    horizon: int,
    max_rounded_paths: int = 30,
    display_graph: bool = False,
)->GCSforBlocks:
    gcs = GCSforBlocks(block_dim, num_blocks, horizon)

    width = 1
    ub = width * 2 * (num_blocks + 1)
    gcs.set_block_width(width)
    gcs.set_ub(ub)

    initial_state = []
    for i in range(gcs.num_modes):
        block_state = [0] * gcs.block_dim
        block_state[0] = width * (2 * i + 1)
        initial_state += block_state
    initial_point = Point(np.array(initial_state))
    final_state = []
    for i in range(gcs.num_modes):
        block_state = [0] * gcs.block_dim
        block_state[-1] = ub - width * (2 * i + 1)
        final_state += block_state
    final_point = Point(np.array(final_state))
    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve(max_rounded_paths=max_rounded_paths)
    gcs.verbose_solution_description()
    if display_graph:
        gcs.display_graph()
    return gcs


def make_some_simple_transparent_tests():
    INFO("--------------------------")
    INFO("Test case: 1D, 3 blocks\n")
    make_simple_transparent_gcs_test(1, 3, 7)
    INFO("--------------------------")
    INFO("Test case: 2D, 3 blocks\n")
    make_simple_transparent_gcs_test(2, 3, 10)
    INFO("--------------------------")
    INFO("Test case: 3D, 5 blocks\n")
    make_simple_transparent_gcs_test(3, 5, 20, 50)


def just_one_simple_test():
    INFO("--------------------------")
    INFO("Test case: 1D, 3 blocks\n")
    make_simple_transparent_gcs_test(2, 3, 15)


just_one_simple_test()
