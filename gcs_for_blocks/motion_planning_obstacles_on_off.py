import typing as T

import numpy as np
import numpy.typing as npt

from .util import timeit, INFO, WARN, ERROR, YAY
from pydrake.solvers import MathematicalProgram
from pydrake.math import le, eq

from .axis_aligned_set_tesselation_2d import (
    AlignedSet,
)
from .tsp_solver import Vertex, Edge

class MotionPlanning:
    def __init__(
        self,
        prog: MathematicalProgram,
        all_vertices:T.Dict[str, Vertex],
        all_edges:T.Dict[str, Edge],
        bounding_box: AlignedSet,
        start_block_pos: T.List[T.Tuple[float, float]],
        target_block_pos: T.List[T.Tuple[float, float]],
        convex_sets: T.Dict[str, AlignedSet],
        moving_block_index: int,
        convex_relaxation=False
    ):
        self.convex_relaxation = convex_relaxation
        self.num_blocks = len(start_block_pos) # type: int
        self.moving_block_index = moving_block_index # type: int
        self.start_block_pos = [np.array(x) for x in start_block_pos] # type: T.List[npt.NDArray]
        self.target_block_pos = [np.array(x) for x in target_block_pos] # type: T.List[npt.NDArray]

        smbi = str(self.moving_block_index) # type: str
        self.start_tsp = "s" + smbi + "_tsp" # type: str
        self.target_tsp = "t" + smbi + "_tsp" # type: str
        self.start_mp = "s" + smbi + "_mp" + smbi # type: str
        self.target_mp = "t" + smbi + "_mp" + smbi # type: str

        self.bounding_box = bounding_box
        self.convex_sets = dict()
        for name in convex_sets:
            new_name = name + "_mp" + str(self.moving_block_index)
            self.convex_sets[new_name] = convex_sets[name].copy()
            self.convex_sets[new_name].name = new_name


        assert len(target_block_pos) == self.num_blocks 

        self.prog = prog
        self.all_vertices = all_vertices
        self.all_edges = all_edges
        self.vertices = dict()
        self.edges = dict()
        self.vertices[self.start_tsp] = self.all_vertices[self.start_tsp]
        self.vertices[self.target_tsp] = self.all_vertices[self.target_tsp]

        self.add_mp_vertices_and_edges()
        self.add_mp_variables_to_prog()
        self.add_mp_constraints_to_prog()

    def add_vertex(self, name: str, value: npt.NDArray = np.array([])):
        assert name not in self.vertices, "Vertex with name " + name + " already exists"
        assert name not in self.all_vertices, "Vertex with name " + name + " already exists in og"
        self.all_vertices[name] = Vertex(name, value)
        self.vertices[name] = self.all_vertices[name]

    def add_edge(self, left_name: str, right_name: str):
        edge_name = left_name + "_" + right_name
        assert edge_name not in self.edges, "Edge " + edge_name + " already exists in new edges"
        assert edge_name not in self.all_edges, "Edge " + edge_name + " already exists in og edges"
        self.all_edges[edge_name] = Edge( self.all_vertices[left_name], self.all_vertices[right_name], edge_name)
        self.edges[edge_name] = self.all_edges[edge_name]
        self.all_vertices[left_name].add_edge_out(edge_name)
        self.all_vertices[right_name].add_edge_in(edge_name)

    def add_mp_vertices_and_edges(self):
        ############################
        # tsp start/target should already be here
        assert self.start_tsp in self.vertices
        assert self.target_tsp in self.vertices

        # add mp vertices
        for aligned_set in self.convex_sets.values():
            self.add_vertex(aligned_set.name)

        ############################
        # add all edges
        # add edge from between tsp portion and mp portion
        self.add_edge(self.start_tsp, self.start_mp)
        self.add_edge(self.target_mp, self.target_tsp)
        # add all edges within the mp portion
        for set1 in self.convex_sets.values():
            for set2 in self.convex_sets.values():
                # no repeats
                if set1 != set2 and set1.share_edge(set2):
                    # add edge between set1 and set2
                    self.add_edge(set1.name, set2.name)

    def add_mp_variables_to_prog(self):
        ###################################
        # add edge variables
        for e in self.edges.values():
            # add flow variable
            if self.convex_relaxation:
                # cotninuous variable, flow between 0 and 1
                e.set_phi(self.prog.NewContinuousVariables(1, "phi_" + e.name)[0])
                self.prog.AddLinearConstraint(e.phi, 0.0, 1.0)
            else:
                e.set_phi(self.prog.NewBinaryVariables(1, "phi_" + e.name)[0])

            # if the edge is not from start
            if e.left.name != self.start_tsp:
                e.set_left_pos(self.prog.NewContinuousVariables(2, "left_pos_" + e.name))
                e.set_right_pos(self.prog.NewContinuousVariables(2, "right_pos_" + e.name))

    def add_mp_constraints_to_prog(self):
        ###################################
        # PER VERTEX
        for v in self.vertices.values():
            # sum_of_y = sum_of_z constraints
            sum_of_y = sum([self.edges[e].left_pos for e in v.edges_out])
            sum_of_z = sum([self.edges[e].right_pos for e in v.edges_in])
            # it's a start node
            if v.name == self.start_tsp:
                continue
            # it's a target node
            elif v.name == self.target_tsp:
                # sum pos out is the target-pos
                block_target_pos = self.target_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint(eq(sum_of_z, block_target_pos))
                # TODO: must add cost to target too
            # it's a start-set node
            elif v.name == self.start_mp:
                # sum pos out is the start-pos
                block_start_pos = self.start_block_pos[self.moving_block_index]
                self.prog.AddLinearConstraint(eq(sum_of_y, block_start_pos))
            # it's any other old node
            else:
                # sum of y equals sum of z
                self.prog.AddLinearConstraint(eq(sum_of_y, sum_of_z))

            # flow in = flow_out constraint
            flow_in = sum([self.edges[e].phi for e in v.edges_in])
            flow_out = sum([self.edges[e].phi for e in v.edges_out])
            if v.name == self.start_tsp:
                self.prog.AddLinearConstraint(flow_out == 1)
            elif v.name == self.target_tsp:
                self.prog.AddLinearConstraint(flow_in == 1)
            else:
                self.prog.AddLinearConstraint(flow_in == flow_out)
                self.prog.AddLinearConstraint(flow_in <= 1)
                self.prog.AddLinearConstraint(flow_out <= 1)

        ###################################
        # PER EDGE
        for e in self.edges.values():
            # for each motion planning edge
            if e.left.name != self.start_tsp and e.right.name != self.target_tsp:
                left_aligned_set = self.convex_sets[e.left.name]
                lA, lb = left_aligned_set.get_perspective_hpolyhedron()
                right_aligned_set = self.convex_sets[e.right.name]
                rA, rb = right_aligned_set.get_perspective_hpolyhedron()
                # left is in the set that corresponds to left
                self.prog.AddLinearConstraint(le(lA @ np.append(e.left_pos, e.phi), lb))
                # right is in the set that corresponds to left and right
                self.prog.AddLinearConstraint(le(lA @ np.append(e.right_pos, e.phi), lb))
                self.prog.AddLinearConstraint(le(rA @ np.append(e.right_pos, e.phi), rb))
            if e.right.name == self.target_tsp:
                # TODO: this should be redundant
                left_aligned_set = self.convex_sets[e.left.name]
                lA, lb = left_aligned_set.get_perspective_hpolyhedron()
                # left is in the set that corresponds to left
                self.prog.AddLinearConstraint(le(lA @ np.append(e.left_pos, e.phi), lb))

            # turning obstacles on and off
            # edge goes into a start obstacle
            if e.left.name != self.start_tsp and e.right.name[0] == "s":
                obstacle_num = int(e.right.name[1:-4])
                x = np.array([self.vertices[self.start_tsp].v[obstacle_num], e.phi])
                A = np.array([[1, 0], [0, -1], [-1, 1]])
                b = np.array([1, 0, 0])
                self.prog.AddLinearConstraint(le(A @ x, b))
            # edge goes into a target obstacle
            if e.right.name != self.target_tsp and e.right.name[0] == "t":
                obstacle_num = int(e.right.name[1:-4])
                if obstacle_num != self.moving_block_index:
                    x = np.array([self.vertices[self.start_tsp].v[obstacle_num], e.phi])
                    A = np.array([[-1, 0], [0, -1], [1, 1]])
                    b = np.array([0, 0, 1])
                    self.prog.AddLinearConstraint(le(A @ x, b))

            # add cost
            if e.left.name != self.start_tsp:
                A = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
                b = np.array([0, 0])
                # TODO: it is annoying that there are a bunch of ~random non-zero edges that have self-cycles
                self.prog.AddL2NormCostUsingConicConstraint(A, b, np.append(e.left_pos, e.right_pos))
                