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



class SetTesselation:
    def __init__(self, options: GCSforAutonomousBlocksOptions):
        self.opt = options
        self.sets_in_dir_representation = self.get_sets_in_dir_representation()

    def get_sets_in_dir_representation(self):
        return all_possible_combinations_of_items(self.opt.dirs, self.opt.set_spec_len)

    def get_constraints_for_direction(self, dir):
        A, b = [], 1
        return A, b

    def add_bounding_box_constraint(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        A = np.vstack((np.eye(self.opt.state_dim), -np.eye(self.opt.state_dim)))
        b = np.hstack((self.opt.ub, -self.opt.lb))
        return A, b





def get_set_from_letters(letters, num_blocks):
    A = []
    b = []
    for letter in letters:
        A_i, b_i = get_constraint_for_direction(letter)
        A += [A_i]
        b += [b_i]
    return np.array(A), np.array(b)



