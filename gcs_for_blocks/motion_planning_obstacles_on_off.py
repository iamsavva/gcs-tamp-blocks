import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram, Solve, L2NormCost, Binding
from pydrake.math import le, eq

from .axis_aligned_set_tesselation_2d import (
    AlignedSet,
    axis_aligned_tesselation,
    locations_to_aligned_sets,
    Box,
)
from .tsp_solver import Vertex, Edge, TSPasGCS


class MotionPlanning(TSPasGCS):
    def __init__(self):
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.vertices = dict()  # type:  T.Dict[str, Vertex]
        self.start = None  # str
        self.target = None  # str

        self.prog = None  # type: MathematicalProgram
        self.solution = None

        self.bounding_box = None
        self.num_blocks = None
        self.convex_sets = None
        self.moving_block_index = None

    def set_bounding_box(self, l:float, r:float, a:float, b:float):
        self.bounding_box = AlignedSet(l=l, r=r, a=a, b=b)
    
    def set_bounding_box_aligned_set(self, aligned_set:AlignedSet):
        self.bounding_box = aligned_set

    def build_the_graph(
        self,
        start_pos: T.List[T.Tuple[float,float]],
        target_pos: T.List[T.Tuple[float,float]],
        visitations: npt.NDArray,
        moving_block_index: int,
        block_width: float = 1.0,
    ):
        # start_block_pos = start_pos[1:]
        # target_block_pos = target_pos[1:]
        start_block_pos = start_pos
        target_block_pos = target_pos
        num_blocks = len(start_block_pos)
        self.start_block_pos = [np.array(x) for x in start_block_pos]
        self.target_block_pos = [np.array(x) for x in target_block_pos]

        self.start = "s" + str(moving_block_index) + "_tsp"
        self.target = "t" + str(moving_block_index) + "_tsp"
        self.start_set = "s" + str(moving_block_index)
        self.target_set = "t" + str(moving_block_index)

        self.num_blocks = len(start_block_pos)
        self.visitations = visitations
        self.moving_block_index = moving_block_index
        
        assert visitations[moving_block_index] == 1
        assert len(target_block_pos) == num_blocks
        assert len(visitations) == num_blocks
        assert self.bounding_box is not None

        # get obstacles
        obstacles = locations_to_aligned_sets(start_block_pos, target_block_pos, block_width)

        # make a tesselation
        self.convex_sets = axis_aligned_tesselation(self.bounding_box.copy(), obstacles)

        ############################
        # add all vertices
        ############################

        # add start vertex
        self.add_vertex(self.start)
        # add target vertex
        self.add_vertex(self.target)
        # add set vertices
        for aligned_set in self.convex_sets.values():
            self.add_vertex(aligned_set.name)

        ############################
        # add all edges
        ############################
        # add edge from si to appropriate set, from ti to appropriate set
        self.add_edge(self.start, self.start_set)
        self.add_edge(self.target_set, self.target)
        # add all edges
        for set1 in self.convex_sets.values():
            for set2 in self.convex_sets.values():
                # no repeats
                if set1 != set2 and set1.share_edge(set2):
                    # add edge between set1 and set2
                    self.add_edge(set1.name, set2.name)


    def build_the_program(self, convex_relaxation=True):
        x = timeit()
        self.prog = MathematicalProgram()
        self.add_variables(convex_relaxation)
        self.add_constraints()
        x.dt("Building the program")
        self.primal_solution = Solve(self.prog)
        x.dt("Solving the program")

        if self.primal_solution.is_success():
            YAY("Optimal primal cost is %.5f" % self.primal_solution.get_optimal_cost())
        else:
            ERROR("PRIMAL SOLVE FAILED!")
            ERROR("Optimal primal cost is %.5f" % self.primal_solution.get_optimal_cost())
            return

        flows = [self.primal_solution.GetSolution(e.phi) for e in self.edges.values()]
        not_tight = np.any(np.logical_and(0.01 < np.array(flows), np.array(flows) < 0.99))
        if not_tight:
            WARN("CONVEX RELAXATION NOT TIGHT")
        else:
            YAY("CONVEX RELAXATION IS TIGHT")

        
    def add_variables(self, convex_relaxation=True):
        ####################################
        # add variables to start and target vertices
        # associated vaiables are visitations, n x 1, each 0 or 1
        self.vertices[self.start].set_v(self.prog.NewContinuousVariables(self.num_blocks, "visit_" + self.start ))
        self.vertices[self.target].set_v(self.prog.NewContinuousVariables(self.num_blocks, "visit_" + self.target ))
        # not adding order here due to irrelevance
        # no other vertex has variables associated with it

        ###################################
        # add edge variables
        for e in self.edges.values():
            # add flow variable
            if convex_relaxation:
                # cotninuous variable, flow between 0 and 1
                e.set_phi(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
                self.prog.AddLinearConstraint(e.phi, 0.0, 1.0)
            else:
                e.set_phi(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])

            # if the edge is not from start
            if e.left.name != self.start:
                # set left and right position variables
                # y is left_pos
                # z is right_pos
                e.set_y(self.prog.NewContinuousVariables(2, "y_" + e.name))
                e.set_z(self.prog.NewContinuousVariables(2, "z_" + e.name))


    def add_constraints(self):
        ###################################
        # visitiations box constraints
        visitation_box = Box(lb=np.zeros(self.num_blocks), ub=np.ones(self.num_blocks), state_dim=self.num_blocks)
        vA, vb = visitation_box.get_hpolyhedron()

        ###################################
        # PER VERTEX
        for v in self.vertices.values():
            # sum_of_y = sum_of_z constraints
            sum_of_y = sum([self.edges[e].y for e in v.edges_out])
            sum_of_z = sum([self.edges[e].z for e in v.edges_in])
            # it's a start node
            if v.name == self.start:
                # add visitations box constraint
                self.prog.AddLinearConstraint(le(vA @ v.v, vb))
                # add visitation equality constraint
                self.prog.AddLinearConstraint(eq(v.v, self.visitations))
            # it's a target node
            elif v.name == self.target:
                # add visitations box constraint
                self.prog.AddLinearConstraint(le(vA @ v.v, vb))
                # add visitation equality constraint to visitations of start
                self.prog.AddLinearConstraint(eq(v.v, self.vertices[self.start].v))
                # sum pos out is the target-pos 
                block_target_pos = self.target_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint( eq( sum_of_z, block_target_pos ) )
                # TODO: must add cost to target too
            # it's a start-set node
            elif v.name == self.start_set:
                # sum pos out is the start-pos
                block_start_pos = self.start_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint( eq( sum_of_y, block_start_pos ) )
            # it's any other old node
            else:
                # sum of y equals sum of z
                self.prog.AddLinearConstraint( eq( sum_of_y, sum_of_z ) )

            # flow in = flow_out constraint
            flow_in = sum([self.edges[e].phi for e in v.edges_in])
            flow_out = sum([self.edges[e].phi for e in v.edges_out])
            if v.name == self.start:
                self.prog.AddLinearConstraint(flow_out == 1)
            elif v.name == self.target:
                self.prog.AddLinearConstraint(flow_in == 1)
            else:
                self.prog.AddLinearConstraint(flow_in == flow_out)
        
        ###################################
        # PER EDGE
        for e in self.edges.values():
            # for each motion planning edge
            if e.left.name != self.start and e.right.name != self.target:
                left_aligned_set = self.convex_sets[e.left.name]
                lA, lb = left_aligned_set.get_perspective_hpolyhedron()
                right_aligned_set = self.convex_sets[e.right.name]
                rA, rb = right_aligned_set.get_perspective_hpolyhedron()
                # left is in the set that corresponds to left
                self.prog.AddLinearConstraint(le(lA @ np.append(e.y, e.phi), lb))
                # right is in the set that corresponds to left and right
                self.prog.AddLinearConstraint(le(lA @ np.append(e.z, e.phi), lb))
                self.prog.AddLinearConstraint(le(rA @ np.append(e.z, e.phi), rb))
            if e.right.name == self.target:
                # TODO: this should be redundant
                left_aligned_set = self.convex_sets[e.left.name]
                lA, lb = left_aligned_set.get_perspective_hpolyhedron()
                # left is in the set that corresponds to left
                self.prog.AddLinearConstraint(le(lA @ np.append(e.y, e.phi), lb))


            # turning obstacles on and off
            # edge goes into a start obstacle
            if e.left.name != self.start and e.right.name[0] == "s":
                # print("start\t", e.name)
                obstacle_num = int(e.right.name[1:])
                x = np.array( [ self.vertices[self.start].v[obstacle_num], e.phi ] )
                A = np.array([[1,0],[0,-1],[-1,1]])
                b = np.array([1,0,0])
                self.prog.AddLinearConstraint(le(A @ x, b))
            # edge goes into a target obstacle
            if e.right.name != self.target and e.right.name[0] == "t":
                # print("target\t", e.name)
                obstacle_num = int(e.right.name[1:])
                if obstacle_num != self.moving_block_index:
                    x = np.array( [ self.vertices[self.start].v[obstacle_num], e.phi ] )
                    A = np.array([[-1,0],[0,-1],[1,1]])
                    b = np.array([0,0,1])
                    self.prog.AddLinearConstraint(le(A @ x, b))

            # add cost
            if e.left.name != self.start:
                A = np.array([[1,0,-1,0],[0,1,0,-1]])
                b = np.array([0,0])
                self.prog.AddL2NormCostUsingConicConstraint(A, b, np.append(e.y, e.z))
            

            


            




