from gcs_for_blocks.tsp_solver import build_block_moving_gcs_tsp, TSPasGCS
import numpy as np


bd = 1
nb = 7
start = np.array([0.0, 1, 2, 3, 4, 5, 6, 7])
target = np.array([2.0, 3, 4, 5, 6, 7, 8, 9])

# nb = 2
# start = np.array([0.0, 1, 2])
# target = np.array([2.0, 3, 4])

# randomness does not break the cycles
# np.random.seed(2)
# start += np.random.normal(0, 0.01, len(start))
# target += np.random.normal(0, 0.01, len(target))
convex_relaxation = True


tsp = build_block_moving_gcs_tsp(start, target, bd, nb)

tsp.solve_primal(convex_relaxation)


# d.build_from_start_and_target(start, target, bd, nb)
# d.build_dual_optimization_program()

# # d.build_primal_optimization_program()
# d.solve_primal(False)
# # d.solve(convex_relaxation)
