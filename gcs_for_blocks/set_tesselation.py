# pyright: reportMissingImports=false
import typing as T

import time
import numpy as np

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    HPolyhedron,
    Iris,
)
from pydrake.common import RandomGenerator  # pylint: disable=import-error

# from .util import WARN, INFO

NUM_DIRS = 4
LETTERS = ["A", "B", "L", "R"]
BLOCK_DIM = 2
BLOCK_WIDTH = 1.0

def inv(letter):
    if letter == "A": return "B"
    if letter == "B": return "A"
    if letter == "L": return "R"
    if letter == "R": return "L"

def all_possible_combinations_of_items(item_set, num_items):
    if num_items == 0:
        return [""]
    result = []
    possible_n_1 = all_possible_combinations_of_items(item_set, num_items-1)
    for item in item_set:
        result += [ item + x for x in possible_n_1 ]
    return result
    

def all_letter_combinations(n:int):
    num_letters_to_define_a_set = n * (n-1) / 2
    return all_possible_combinations_of_items(LETTERS, num_letters_to_define_a_set)

def get_constraint_for_direction(dir, num_blocks):
    state_dim = num_blocks * BLOCK_DIM
    A, b = [], 1
    return A, b
    # A = []
    # b = []
    # if dir == "A":





def get_set_from_letters(letters, num_blocks):
    A = []
    b = []
    for letter in letters:
        A_i, b_i = get_constraint_for_direction(letter)
        A += [A_i]
        b += [b_i]
    return np.array(A), np.array(b)
        
        



