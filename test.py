import numpy as np
from gcs import GCSforBlocks
from pydrake.geometry.optimization import Point

gcs = GCSforBlocks()
initial_state = Point(np.array([0, 2, 4]))
final_state = Point(np.array([0, 5, 7]))
gcs.build_the_graph(initial_state, 0, final_state, 0)
gcs.solve(False)
