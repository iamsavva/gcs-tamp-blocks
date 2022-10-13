# import itertools
# from dataclasses import dataclass
# from typing import List, Optional, Tuple
import typing as T

import numpy as np
import numpy.typing as npt

# import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import Point, GraphOfConvexSets, HPolyhedron
from pydrake.math import eq
from pydrake.solvers import (
    Binding,
    Cost,
    L1NormCost,
    L2NormCost,
    MathematicalProgramResult,
)
from tqdm import tqdm

# from geometry.contact import CollisionPair, calc_intersection_of_contact_modes
# from geometry.polyhedron import PolyhedronFormulator


class GCSforBlocks:
    block_dim: int  # number of dimensions that describe the block world
    num_blocks: int  # number of blocks
    horizon: int  # number of mode swithes
    lb: npt.NDArray  # lower bound on box that bounds the operational space
    ub: npt.NDArray  # upper bound on box bounds the operational space
    width: float = 1.0  # block width
    delta: float = width / 2  # half block width

    @property
    def num_modes(self) -> int:  # pylint: disable=missing-function-docstring
        return self.num_blocks + 1

    @property
    def state_dim(self) -> int:  # pylint: disable=missing-function-docstring
        return self.num_modes * self.block_dim

    def __init__(self, block_dim=1, num_blocks=2, horizon=5):
        self.block_dim = block_dim
        self.num_blocks = num_blocks
        self.horizon = horizon

        self.lb = np.zeros(self.state_dim)
        self.ub = 10.0 * np.ones(self.state_dim)

        self.gcs = GraphOfConvexSets()

        self.horizon_layers = []

        self.mode_edges = self.define_edges_between_modes()

    def define_edges_between_modes(self):
        """
        Return a matrix that represents edges in a directed graph of modes.
        For this simple example, the matrix is hand-built.
        When IRIS is used, sets must be A -- clustered (TODO: do they?),
        B -- connectivity checked and defined automatically.
        """
        mat = np.zeros((self.num_modes, self.num_modes))
        # mode 0 is connected to any other mode
        mat[0, :] = np.ones(self.num_modes)
        # mode k is connected only to 0;
        # TODO: do i need k to k transitions? don't think so
        mat[:, 0] = np.ones(self.num_modes)
        return mat

    # def get_v_in(self, )

    def add_start_node(self, initial_state: Point, initial_mode: int):
        """
        Adds start node to the graph.
        TODO: checking that start hasn't been added already + function to change the start configuration
        """
        # check that point belongs to the corresponding mode
        mode_set = self.get_convex_set_for_mode(initial_mode)
        assert mode_set.PointInSet(initial_state.x())
        # add node to the graph
        assert len(self.horizon_layers) == 0
        name = "start"
        self.gcs.AddVertex(initial_state, name)
        self.horizon_layers += [name]

    def add_target_node(self, final_state, final_mode):
        """
        Adds target node to the graph.
        TODO: checking that target hasn't been added already + function to change the target configuration
        TODO: add edges to nodes in a previous layer
        """
        # check that point belongs to the corresponding mode
        mode_set = self.get_convex_set_for_mode(final_mode)
        assert mode_set.PointInSet(final_state.x())
        # add node to the graph
        name = "target"
        self.gcs.AddVertex(final_state, name)
        self.horizon_layers += [name]
        # TODO: must add edges to the nodes in the previous layer, as well as corresponding constraints

    def get_vertex_name(self, layer, mode):
        return "M_" + str(layer) + "_" + str(mode)

    def add_nodes_for_next_layer(self, layer):
        for mode in range(self.num_modes):
            self.gcs.AddVertex(
                self.get_convex_set_for_mode(mode), self.get_vertex_name(layer, mode)
            )

    def get_convex_set_for_mode(self, k: int) -> HPolyhedron:
        """
        Returns polyhedron:
        lb <= x <= ub
        x_0 = x_k
        (last constrained dropped if k = 0)
        """
        assert k < self.num_modes
        A = np.vstack((-np.eye(self.state_dim), np.eye(self.state_dim)))
        b = np.hstack((-self.lb, self.ub))
        if k == 0:
            return HPolyhedron(A, b)
        else:
            eq_con = np.zeros((self.block_dim, self.state_dim))
            eq_con[:, 0 : self.block_dim] = np.eye(self.block_dim)
            eq_con[:, k * self.block_dim : (k + 1) * self.block_dim] = -np.eye(
                self.block_dim
            )
            A = np.vstack((A, eq, -eq))
            b = np.hstack((b, np.zeros(self.block_dim), np.zeros(self.block_dim)))
            return HPolyhedron(A, b)

    def get_constraint_for_orbit_of_mode(self, k: int):
        """
        Delta x_i is zero for any i != 0, k.
        """
        assert k < self.num_modes
        A = np.eye(self.state_dim)
        A[0 : self.block_dim, 0 : self.block_dim] = np.zeros(
            (self.block_dim, self.block_dim)
        )
        A[
            k * self.block_dim : (k + 1) * self.block_dim,
            k * self.block_dim : (k + 1) * self.block_dim,
        ] = np.zeros((self.block_dim, self.block_dim))
        b = np.zeros(self.state_dim)
        # TODO: what am I returning here
