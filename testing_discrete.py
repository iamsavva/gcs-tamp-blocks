from gcs_for_blocks.discrete_network_flow_solver import DiscreteNetworkFlowGraph
import numpy as np

d = DiscreteNetworkFlowGraph()

bd = 1
nb = 7
start = np.array([0,1,2,3,4,5,6,7])
target = np.array([21,20,3,30,9, 59, 31, 0])

d.build_from_start_and_target(start,target, bd, nb)
