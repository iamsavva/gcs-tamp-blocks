from pydrake.geometry.optimization import Point
import numpy as np

from gcs import GCSforBlocks
from util import INFO
from pydrake.geometry.optimization import Point, GraphOfConvexSets, HPolyhedron, Iris


def make_simple_swap_two(horizon=10, max_rounded_paths=200, use_convex_relaxation=True):
    num_blocks = 2
    block_dim = 2

    gcs = GCSforBlocks(block_dim, num_blocks, horizon)
    gcs.no_cycles = False
    gcs.problem_complexity = "obstacles"

    width = 1
    ub = 2
    gcs.set_block_width(width)
    gcs.set_ub(ub)

    # initial_state = []
    initial_state = [1, 0, 1, 1, 1, 2]
    # for i in range(gcs.num_modes):
    #     block_state = [0] * gcs.block_dim
    #     block_state[0] = width * (2 * i + 1)
    #     initial_state += block_state
    initial_point = Point(np.array(initial_state))
    final_state = [1, 0, 1, 2, 1, 1]
    # for i in range(gcs.num_modes):
    #     block_state = [0] * gcs.block_dim
    #     block_state[-1] = ub - width * (2 * i + 1)
    #     final_state += block_state
    final_point = Point(np.array(final_state))
    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve(
        use_convex_relaxation=use_convex_relaxation, max_rounded_paths=max_rounded_paths
    )
    gcs.verbose_solution_description()
    # if display_graph:
    #     gcs.display_graph()
    return gcs, ub, final_state


def make_simple_swap_three(
    horizon=10, max_rounded_paths=200, use_convex_relaxation=True
):
    num_blocks = 3
    block_dim = 2

    gcs = GCSforBlocks(block_dim, num_blocks, horizon)
    gcs.no_cycles = True
    gcs.problem_complexity = "obstacles"

    width = 1
    ub = 3
    gcs.set_block_width(width)
    gcs.set_ub(ub)

    # initial_state = []
    initial_state = [0, 0, 1.5, 1, 1.5, 2, 1.5, 3]
    # for i in range(gcs.num_modes):
    #     block_state = [0] * gcs.block_dim
    #     block_state[0] = width * (2 * i + 1)
    #     initial_state += block_state
    initial_point = Point(np.array(initial_state))
    final_state = [0, 0, 1.5, 3, 2.5, 3, 1.5, 2]
    # for i in range(gcs.num_modes):
    #     block_state = [0] * gcs.block_dim
    #     block_state[-1] = ub - width * (2 * i + 1)
    #     final_state += block_state
    final_point = Point(np.array(final_state))
    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve(
        use_convex_relaxation=use_convex_relaxation, max_rounded_paths=max_rounded_paths
    )
    gcs.verbose_solution_description()
    # if display_graph:
    #     gcs.display_graph()
    return gcs, ub, final_state


def make_simple_set_based(
    horizon: int,
    use_convex_relaxation=True,
    max_rounded_paths: int = 30,
    display_graph: bool = False,
) -> GCSforBlocks:
    block_dim = 2
    num_blocks = 2
    gcs = GCSforBlocks(block_dim, num_blocks, horizon)
    width = 1
    ub = 3
    gcs.set_block_width(width)
    gcs.set_ub(ub)
    gcs.no_cycles = False
    gcs.problem_complexity = "obstacles"

    initial_state = [0, 0, 1.5, 0, 1.5, 1.5]

    initial_point = Point(np.array(initial_state))
    A = np.zeros((2, 6))
    A[:, 4:6] = np.eye(2)
    final = np.array([1.5, 3])
    b = final
    A = np.vstack((A, -A))
    b = np.hstack((b, -b))
    final_point = HPolyhedron(A, b)

    # final_point = Point(np.array(final_state))
    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve(
        use_convex_relaxation=use_convex_relaxation, max_rounded_paths=max_rounded_paths
    )
    gcs.verbose_solution_description()
    if display_graph:
        gcs.display_graph()
    return gcs, ub, final


def make_simple_transparent_gcs_test(
    block_dim: int,
    num_blocks: int,
    horizon: int,
    use_convex_relaxation=True,
    max_rounded_paths: int = 30,
    display_graph: bool = False,
) -> GCSforBlocks:
    gcs = GCSforBlocks(block_dim, num_blocks, horizon)
    width = 1
    scaling = 0.5
    ub = scaling * width * 2 * (num_blocks + 1)
    gcs.set_block_width(width)
    gcs.set_ub(ub)
    gcs.no_cycles = False
    gcs.problem_complexity = "transparent-no-obstacles"

    initial_state = []
    for i in range(gcs.num_modes):
        block_state = [0] * gcs.block_dim
        block_state[0] = scaling * width * (2 * i + 1)
        initial_state += block_state
    initial_point = Point(np.array(initial_state))
    final_state = []
    for i in range(gcs.num_modes):
        block_state = [ub] * gcs.block_dim
        block_state[0] = ub - scaling * width * (2 * i + 1)
        final_state += block_state
    final_point = Point(np.array(final_state))
    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve(
        use_convex_relaxation=use_convex_relaxation, max_rounded_paths=max_rounded_paths
    )
    gcs.verbose_solution_description()
    if display_graph:
        gcs.display_graph()
    return gcs, ub, final_state


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


# just_one_simple_test()
