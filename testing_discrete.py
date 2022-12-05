from gcs_for_blocks.tsp_solver import build_block_moving_gcs_tsp, TSPasGCS
import numpy as np


a = np.array([1, 2])
b = 3
print(np.append(a, b))

bd = 1
nb = 9

# why do cycles occur?
# np.random.seed(1)
# some random seeds break symmetries; figure out what those symmetries are!!!
# start = np.random.uniform(0, 50, nb+1)
# target = np.random.uniform(0, 50, nb+1)

start = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + np.random.normal(0, 0.001, nb + 1)
target = np.array([2.0, 2, 4, 5, 6, 7, 30, 9, 11, 12]) + np.random.normal(0, 0.001, nb + 1)

# start = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 10, 20]) + np.random.normal(0,0.001, nb+1)
# target = np.array([2.0, 2, 4, 5, 6, 7, 30, 9, 11, 12]) + np.random.normal(0,0.001, nb+1)

# nb = 4
# start = np.array([0.0, 1, 5, 6, 7])
# target = np.array([2.0, 3, 7, 8, 9])

# nb = 4
# start = np.array([0.0, 1, 2, 3, 4])
# target = np.array([2.0, 3, 4, 5, 6])

# nb = 2
# start = np.array([0.0, 1, 2])
# target = np.array([2.0, 3, 4])

# randomness does not break the cycles
# np.random.seed(2)
# start += np.random.normal(0, 0.01, len(start))
# target += np.random.normal(0, 0.01, len(target))
# convex_relaxation = False
convex_relaxation = True


tsp = build_block_moving_gcs_tsp(start, target, bd, nb)

tsp.solve_primal(convex_relaxation)


# d.build_from_start_and_target(start, target, bd, nb)
# d.build_dual_optimization_program()

# # d.build_primal_optimization_program()
# d.solve_primal(False)
# # d.solve(convex_relaxation)
