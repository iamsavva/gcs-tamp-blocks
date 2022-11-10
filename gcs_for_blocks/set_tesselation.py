#!/usr/bin/env python3
# pyright: reportMissingImports=false
import typing as T

import time
import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    HPolyhedron,
    Iris,
)

from pydrake.common import RandomGenerator  # pylint: disable=import-error

from .gcs_options import GCSforAutonomousBlocksOptions
from .util import WARN, INFO, all_possible_combinations_of_items

from tqdm import tqdm

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    HPolyhedron,
    ConvexSet,
)


class SetTesselation:
    def __init__(self, options: GCSforAutonomousBlocksOptions):
        self.opt = options
        self.sets_in_dir_representation = self.get_sets_in_dir_representation()

        t0 = time.time()
        self.index2block_rel = dict()  # T.Dict[int, (int,int)]
        self.make_index_to_block_relation()
        t1 = time.time()
        print(t1 - t0)

        self.dir2set = dict()  # T.Dict[str, HPolyhedron]
        self.generate_sets()
        t2 = time.time()
        print(t2 - t1)

    def get_sets_in_dir_representation(self):
        return all_possible_combinations_of_items(self.opt.dirs, self.opt.set_spec_len)

    def generate_sets(self):
        for i in tqdm(range(len(self.sets_in_dir_representation))):
            dir_rep = self.sets_in_dir_representation[i]
            # get set
            set_for_dir_rep = self.gen_set_from_dir_rep(dir_rep)
            # DO NOT reduce iequalities, some of these sets are empty
            # reducing inequalities is also extremely time consuming
            # set_for_dir_rep = set_for_dir_rep.ReduceInequalities()

            # check that it's non-empty
            try:
                set_for_dir_rep.ChebyshevCenter()
                self.dir2set[dir_rep] = set_for_dir_rep
            except RuntimeError:
                # this set is empty
                continue

    def gen_set_from_dir_rep(self, dir_rep: str):
        A, b = self.get_bounding_box_constraint()
        for index, dir in enumerate(dir_rep):
            i, j = self.index2block_rel[index]
            A_dir, b_dir = self.get_constraints_for_direction(dir, i, j)
            A = np.vstack((A, A_dir))
            b = np.hstack((b, b_dir))
        return HPolyhedron(A, b)

    def get_constraints_for_direction(self, dir: str, i, j):
        w = self.opt.block_width
        bd = self.opt.block_dim
        A = np.zeros((2, self.opt.state_dim))
        if dir == "A":
            A[0, j * bd], A[0, i * bd] = 1, -1
            A[1, j * bd + 1], A[1, i * bd + 1] = 1, -1
            b = np.array([-w, w])
        elif dir == "B":
            A[0, i * bd], A[0, j * bd] = 1, -1
            A[1, i * bd + 1], A[1, j * bd + 1] = 1, -1
            b = np.array([-w, w])
        elif dir == "L":
            A[0, i * bd], A[0, j * bd] = 1, -1
            A[1, j * bd + 1], A[1, i * bd + 1] = 1, -1
            b = np.array([w, -w])
        elif dir == "R":
            A[0, j * bd], A[0, i * bd] = 1, -1
            A[1, i * bd + 1], A[1, j * bd + 1] = 1, -1
            b = np.array([w, -w])
        return A, b

    def get_bounding_box_constraint(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        A = np.vstack((np.eye(self.opt.state_dim), -np.eye(self.opt.state_dim)))
        b = np.hstack((self.opt.ub, -self.opt.lb))
        return A, b

    def make_index_to_block_relation(self):
        """
        01 02 03 .. 0n-1
        12 13 14    1n-1
        ...
        n-2 n-1
        """
        st = [0, 1]
        index = 0
        while index < self.opt.set_spec_len:
            self.index2block_rel[index] = (st[0], st[1])
            index += 1
            st[1] += 1
            if st[1] == self.opt.num_blocks:
                st[0] += 1
                st[1] = st[0] + 1
        assert st == [self.opt.num_blocks - 1, self.opt.num_blocks], "checking my math"
