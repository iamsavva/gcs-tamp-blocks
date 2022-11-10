from gcs_for_blocks.set_tesselation_2d import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions
from gcs_for_blocks.gcs_auto_blocks import GCSAutonomousBlocks

import numpy as np

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    HPolyhedron,
    ConvexSet,
)

from draw_2d import Draw2DSolution

if __name__ == "__main__":
    nb = 2
    ubf = 2.0
    start_point = Point( np.array([1,1, 1,2]))
    target_point = Point(np.array([1,2, 1,1]))

    nb = 3
    ubf = 4.0
    start_point = Point(np.array([1,1, 1,2, 1,3]))
    target_point = Point(np.array([3,3, 3,1, 3,2]))

    # nb = 4
    # ubf = 4.0
    # start_point = Point(np.array([1,1, 1,2, 1,3, 1,4]))
    # target_point = Point(np.array([3,4, 3,3, 3,2, 3,1]))

    options = GCSforAutonomousBlocksOptions(nb, ubf = ubf)
    options.use_convex_relaxation = True
    options.max_rounded_paths = 10

    gcs = GCSAutonomousBlocks(options)
    gcs.build_the_graph(start_point, target_point)
    gcs.solve(show_graph=True)
    modes, vertices = gcs.get_solution_path()

    drawer = Draw2DSolution(nb, gcs.opt.ub, modes, vertices, target_point.x(), fast = False, no_arm = True)
    drawer.draw_solution_no_arm()


