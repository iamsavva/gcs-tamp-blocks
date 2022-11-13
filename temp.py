from gcs_for_blocks.set_tesselation_2d import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions
from gcs_for_blocks.gcs_auto_blocks import GCSAutonomousBlocks
from gcs_for_blocks.util import timeit, INFO

import numpy as np

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    HPolyhedron,
    ConvexSet,
)

from draw_2d import Draw2DSolution

if __name__ == "__main__":

    # tricky case to screw up solution optimality
    # nb = 2
    # ubf = 4.0
    # start_point = Point( np.array([2,2, 0,1-0.01]))
    # target_point = Point(np.array([2,2, 3+0.01,3-0.01]))

    # nb = 2
    # ubf = 2.0
    # start_point = Point( np.array([1,1, 1,2]))
    # target_point = Point(np.array([1,2, 1,1]))

    # nb = 3
    # ubf = 4.0
    # start_point = Point(np.array([1,1, 1,2, 1,3]))
    # target_point = Point(np.array([3,3, 3,1, 3,2]))

    # # 5.31 5.27 

    nb = 4
    ubf = 4.0
    start_point = Point(np.array([1,1, 1,2, 1,3, 1,4]))
    target_point = Point(np.array([3,4, 3,3, 3,2, 3,1]))

    options = GCSforAutonomousBlocksOptions(nb, ubf = ubf)
    options.use_convex_relaxation = True
    options.max_rounded_paths = 0
    options.edge_gen = "binary_tree_down" # binary_tree_down
    options.symmetric_set_def = True

    x = timeit()
    gcs = GCSAutonomousBlocks(options)
    # x.dt("initing")
    gcs.build_the_graph_simple(start_point, target_point)
    INFO("Initial number of v/e is ", len(gcs.gcs.Vertices()), len(gcs.gcs.Edges()))
    # gcs.display_graph()
    # x.dt("building")
    gcs.solve_plot_sparse()
    INFO("Number of used v/e is ", len(gcs.gcs.Vertices()), len(gcs.gcs.Edges()))
    # gcs.solve(show_graph=False)
    # x.dt("solving")
    # modes, vertices = gcs.get_solution_path()
    # print(modes)
    # print(modes)
    # print(vertices)

    # drawer = Draw2DSolution(nb, gcs.opt.ub, modes, vertices, target_point.x(), fast = False, no_arm = True)
    # drawer.draw_solution_no_arm()


