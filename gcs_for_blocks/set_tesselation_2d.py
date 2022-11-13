#!/usr/bin/env python3
# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import HPolyhedron

from .gcs_options import GCSforAutonomousBlocksOptions
from .util import WARN, INFO, all_possible_combinations_of_items, timeit, ChebyshevCenter

from tqdm import tqdm

class SetTesselation:
    def __init__(self, options: GCSforAutonomousBlocksOptions):
        self.opt = options
        self.sets_in_dir_representation = self.get_sets_in_dir_representation()

        self.index2block_rel = dict()  # T.Dict[int, (int,int)]
        self.make_index_to_block_relation()

        self.dir2set = dict()  # T.Dict[str, HPolyhedron]
        self.generate_sets()

    def get_sets_in_dir_representation(self):
        return all_possible_combinations_of_items(self.opt.dirs, self.opt.set_spec_len)

    def generate_sets(self):
        for i in tqdm(range(len(self.sets_in_dir_representation)), "Set generation"):
            dir_rep = self.sets_in_dir_representation[i]
            # get set
            set_for_dir_rep = self.gen_set_from_dir_rep(dir_rep)
            # DO NOT reduce iequalities, some of these sets are empty
            # reducing inequalities is also extremely time consuming
            # set_for_dir_rep = set_for_dir_rep.ReduceInequalities()

            # check that it's non-empty
            solved, x, r = ChebyshevCenter(set_for_dir_rep)
            if solved and r >= 0.00001:
                self.dir2set[dir_rep] = set_for_dir_rep
            

    def gen_set_from_dir_rep(self, dir_rep: str):
        A, b = self.get_bounding_box_constraint()
        for index, dir in enumerate(dir_rep):
            i, j = self.index2block_rel[index]
            A_dir, b_dir = self.get_constraints_for_direction(dir, i, j)
            A = np.vstack((A, A_dir))
            b = np.hstack((b, b_dir))
        return HPolyhedron(A, b)

    def get_constraints_for_direction(self, dir, i, j):
        if self.opt.symmetric_set_def:
            return self.get_constraints_for_direction_sym(dir,i,j)
        else:
            return self.get_constraints_for_direction_asym(dir,i,j)

    def get_constraints_for_direction_asym(self, dir: str, i, j):
        w = self.opt.block_width
        bd = self.opt.block_dim
        A = np.zeros((2, self.opt.state_dim))
        if dir == "A":
            A[0, j * bd], A[0, i * bd] = 1, -1
            A[1, j * bd + 1], A[1, i * bd + 1] = 1, -1
            b = np.array([w, -w])
        elif dir == "B":
            A[0, i * bd], A[0, j * bd] = 1, -1
            A[1, i * bd + 1], A[1, j * bd + 1] = 1, -1
            b = np.array([w, -w])
        elif dir == "L":
            A[0, i * bd], A[0, j * bd] = 1, -1
            A[1, j * bd + 1], A[1, i * bd + 1] = 1, -1
            b = np.array([-w, w])
        elif dir == "R":
            A[0, j * bd], A[0, i * bd] = 1, -1
            A[1, i * bd + 1], A[1, j * bd + 1] = 1, -1
            b = np.array([-w, w])
        return A, b

    def get_constraints_for_direction_sym(self, dir: str, i, j):
        w = self.opt.block_width
        bd = self.opt.block_dim
        sd = self.opt.state_dim
        xi, yi = i * bd, i * bd + 1
        xj, yj = j * bd, j * bd + 1
        a0, a1, a2 = np.zeros(sd), np.zeros(sd), np.zeros(sd)
        if dir == "L":
            a0[xi], a0[yi], a0[xj], a0[yj] = 1, -1, -1, 1
            a1[xi], a1[yi], a1[xj], a1[yj] = 1, 1, -1, -1
            a2[xi], a2[yi], a2[xj], a2[yj] = 1, 0, -1, 0
        elif dir == "A":
            a0[xi], a0[yi], a0[xj], a0[yj] = 1, -1, -1, 1
            a1[xi], a1[yi], a1[xj], a1[yj] = -1, -1, 1, 1
            a2[xi], a2[yi], a2[xj], a2[yj] = 0, -1, 0, 1
        elif dir == "R":
            a0[xi], a0[yi], a0[xj], a0[yj] = -1, 1, 1, -1
            a1[xi], a1[yi], a1[xj], a1[yj] = -1, -1, 1, 1
            a2[xi], a2[yi], a2[xj], a2[yj] = -1, 0, 1, 0
        elif dir == "B":
            a0[xi], a0[yi], a0[xj], a0[yj] = -1, 1, 1, -1
            a1[xi], a1[yi], a1[xj], a1[yj] = 1, 1, -1, -1
            a2[xi], a2[yi], a2[xj], a2[yj] = 0, 1, 0, -1
        A = np.vstack((a0, a1, a2))
        b = np.array([0, 0, -w])
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

    def construct_dir_representation_from_point(self, point: npt.NDArray)->str:
        dir_representation = ""
        for index in range( self.opt.set_spec_len ):
            i, j = self.index2block_rel[index]
            for dir in self.opt.dirs:
                A, b = self.get_constraints_for_direction(dir, i, j)
                if np.all(A.dot(point) <= b):
                    dir_representation += dir
                    break
        assert len(dir_representation) == self.opt.set_spec_len
        return dir_representation


    def get_1_step_neighbours(self, dir:str):
        assert len(dir) == self.opt.set_spec_len, "inappropriate dir: " + dir
        ldir = list(dir)
        nbhd = []
        for i in range(len(dir)):
            for j in range(self.opt.num_dirs-1):
                ldir[i] = self.opt.dir_iter(ldir[i])
                if self.opt.dir_inv(dir[i]) != ldir[i] and ''.join(ldir) in self.dir2set:
                    nbhd += [''.join(ldir)]
            ldir[i] = self.opt.dir_iter(ldir[i])
        return nbhd

    def get_useful_1_step_neighbours(self, dir:str, target:str):
        assert len(dir) == self.opt.set_spec_len, "inappropriate dir: " + dir
        assert len(target) == self.opt.set_spec_len, "inappropriate target: " + target

        nbhd = []
        for i in range(len(dir)):
            if dir[i] == target[i]:
                continue
            elif target[i] in self.opt.dir_nbhd[dir[i]]:
                nbhd += [dir[:i] + target[i] + dir[i+1:]]
            else:
                for let in self.opt.dir_nbhd[dir[i]]:
                    nbhd += [dir[:i] + let + dir[i+1:]]
        return nbhd




        

