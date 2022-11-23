from gcs_for_blocks.set_tesselation_2d import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions
from gcs_for_blocks.gcs_auto_blocks import GCSAutonomousBlocks
from gcs_for_blocks.util import timeit, INFO
from gcs_for_blocks.hierarchical_gcs_ab import HierarchicalGraph, HierarchicalGCSAB

import numpy as np

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    HPolyhedron,
    ConvexSet,
)


from draw_2d import Draw2DSolution

if __name__ == "__main__":
    # f = float('inf')
    # print(f == float('inf'))
    # nb = 2
    # ubf = 2.0
    # start_point = Point( np.array([1,1, 1,2]))
    # target_point = Point(np.array([1,2, 1,1]))

    # nb = 3
    # ubf = 4.0
    # start_point = Point(np.array([1, 1, 1, 2, 1, 3]))
    # target_point = Point(np.array([2, 3, 2, 2, 2, 1]))

    nb = 4
    ubf = 4.0
    start_point = Point(np.array([1, 1, 1, 2, 1, 3, 1, 4]))
    target_point = Point(np.array([3, 4, 3, 3, 3, 2, 3, 1]))

    options = GCSforAutonomousBlocksOptions(nb, ubf=ubf)
    hg = HierarchicalGCSAB(options)
    hg.solve(start_point, target_point)


# if __name__ == "__main__":
#     # tricky case to screw up solution optimality
#     # nb = 2
#     # ubf = 4.0
#     # start_point = Point( np.array([2,2, 0,1-0.01]))
#     # target_point = Point(np.array([2,2, 3+0.01,3-0.01]))

#     # nb = 2
#     # ubf = 2.0
#     # start_point = Point( np.array([1,1, 1,2]))
#     # target_point = Point(np.array([1,2, 1,1]))

#     nb = 3
#     ubf = 4.0
#     start_point = Point(np.array([1, 1, 1, 2, 1, 3]))
#     target_point = Point(np.array([3, 3, 3, 1, 3, 2]))

#     # # 5.31 5.27

#     nb = 4
#     ubf = 4.0
#     start_point = Point(np.array([1,1, 1,2, 1,3, 1,4]))
#     target_point = Point(np.array([3,4, 3,3, 3,2, 3,1]))

#     options = GCSforAutonomousBlocksOptions(nb, ubf=ubf)
#     options.use_convex_relaxation = True
#     options.max_rounded_paths = 30
#     options.problem_complexity = "collision-free-all-moving"
#     options.edge_gen = "binary_tree_down"  # binary_tree_down
#     options.symmetric_set_def = True
#     options.rounding_seed = 1
#     options.custom_rounding_paths = 0

#     x = timeit()
#     gcs = GCSAutonomousBlocks(options)
#     gcs.build_the_graph_simple(start_point, target_point)
#     gcs.solve(show_graph=False, verbose=True)
#     costs = np.array([cost for (_, cost) in gcs.get_solution_path()])
#     print(costs)

#     # INFO("Initial number of v/e is ", len(gcs.gcs.Vertices()), len(gcs.gcs.Edges()))
#     # gcs.display_graph()
#     # x.dt("building")
#     # gcs.solve_plot_sparse()
#     # INFO("Number of used v/e is ", len(gcs.gcs.Vertices()), len(gcs.gcs.Edges()))
#     # gcs.solve(show_graph=False)
#     # x.dt("solving")
#     # modes, vertices = gcs.get_solution_path()
#     # print(modes)
#     # print(modes)
#     # print(vertices)

#     # drawer = Draw2DSolution(nb, gcs.opt.ub, modes, vertices, target_point.x(), fast = False, no_arm = True)
#     # drawer.draw_solution_no_arm()
