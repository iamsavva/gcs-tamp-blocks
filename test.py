# pyright: reportMissingImports=false
import typing as T

from pydrake.geometry.optimization import Point  # pylint: disable=import-error
import numpy as np
import numpy.typing as npt

from gcs_for_blocks.gcs import GCSforBlocks
from gcs_for_blocks.gcs_in_out import GCSforBlocksOneInOneOut
from gcs_for_blocks.gcs_options import GCSforBlocksOptions
from gcs_for_blocks.util import INFO

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
)


def make_simple_obstacle_swap_two_in_out(
    use_convex_relaxation: bool = False, max_rounded_paths: int = 100
) -> T.Tuple[GCSforBlocksOneInOneOut, npt.NDArray, T.List]:
    INFO("--------------------------")
    INFO("Test case: 2D, Obstacles, 2 blocks IN OUT\n")
    options = GCSforBlocksOptions(num_blocks=2, block_dim=2, horizon=10)
    options.problem_complexity = "obstacles"
    options.use_convex_relaxation = use_convex_relaxation
    options.max_rounded_paths = max_rounded_paths

    gcs = GCSforBlocksOneInOneOut(options)
    width = 1.0
    ub = np.array([2, 2])
    gcs.set_block_width(width)
    gcs.set_ub(ub)

    initial_state = [1, 0, 1, 1, 1, 2]
    initial_point = Point(np.array(initial_state))
    final_state = [1, 0, 1, 2, 1, 1]
    final_point = Point(np.array(final_state))

    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve()
    try:
        gcs.verbose_solution_description()
    except:
        pass
    return gcs, ub, final_state


def make_simple_obstacle_swap_two(
    use_convex_relaxation: bool = False, max_rounded_paths: int = 100
) -> T.Tuple[GCSforBlocks, npt.NDArray, T.List]:
    INFO("--------------------------")
    INFO("Test case: 2D, Obstacles, 2 blocks\n")
    options = GCSforBlocksOptions(num_blocks=2, block_dim=2, horizon=10)
    options.problem_complexity = "obstacles"
    options.use_convex_relaxation = use_convex_relaxation
    options.max_rounded_paths = max_rounded_paths

    gcs = GCSforBlocks(options)
    width = 1.0
    ub = np.array([2, 2])
    gcs.set_block_width(width)
    gcs.set_ub(ub)

    initial_state = [1, 0, 1, 1, 1, 2]
    initial_point = Point(np.array(initial_state))
    final_state = [1, 0, 1, 2, 1, 1]
    final_point = Point(np.array(final_state))

    gcs.build_the_graph(initial_point, 0, final_point, 0)
    gcs.solve()
    try:
        gcs.verbose_solution_description()
    except:
        pass
    return gcs, ub, final_state


def make_simple_transparent_gcs_test(
    block_dim: int,
    num_blocks: int,
    horizon: int,
    use_convex_relaxation: bool = True,
    max_rounded_paths: int = 100,
    display_graph: bool = False,
) -> T.Tuple[GCSforBlocks, npt.NDArray, T.List]:
    options = GCSforBlocksOptions(
        block_dim=block_dim, num_blocks=num_blocks, horizon=horizon
    )
    options.use_convex_relaxation = use_convex_relaxation
    options.max_rounded_paths = max_rounded_paths
    options.problem_complexity = "transparent-no-obstacles"

    gcs = GCSforBlocks(options)

    width = 1
    scaling = 0.5
    ub_float = scaling * width * 2 * (num_blocks + 1)
    ub = np.tile((np.array(ub_float)), block_dim)
    gcs.set_block_width(width)
    gcs.set_ub(ub)

    # make initial state
    initial_state = []
    for i in range(options.num_modes):
        block_state = [0] * options.block_dim
        block_state[0] = scaling * width * (2 * i + 1)  # type: ignore
        initial_state += block_state
    initial_point = Point(np.array(initial_state))
    # make final state
    final_state = []
    for i in range(options.num_modes):
        block_state = [ub_float] * options.block_dim
        block_state[0] = ub_float - scaling * width * (2 * i + 1)
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
    make_simple_transparent_gcs_test(2, 3, 7)
    INFO("--------------------------")
    INFO("Test case: 3D, 5 blocks\n")
    make_simple_transparent_gcs_test(3, 5, 18)


if __name__ == "__main__":
    make_simple_obstacle_swap_two()
    make_some_simple_transparent_tests()
