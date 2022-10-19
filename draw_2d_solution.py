import typing as T

import numpy as np
import numpy.typing as npt

from gcs import GCSforBlocks
from test import make_simple_transparent_gcs_test

try:
  from tkinter import Tk, Canvas, Toplevel
except ImportError:
  from Tkinter import Tk, Canvas, Toplevel

import colorsys
from sim.util.utils import user_input
import sys

BLOCK_COLORS = ['#E3B5A4', '#E8D6CB', '#C3DFE0', '#F6E4F6', '#F4F4F4']
ARM_COLOR = '#621940'
ARM_NOT_EMPTY_COLOR = '#000000'

TEXT_COLOR = '#0B032D'
BLACK = '#0B032D'
BACKGROUND = '#F5E9E2'

ARM_SIZE = 20
BLOCK_SIZE = 25
GRID_SIZE = 30

class Object:
  def __init__(self, state: npt.NDArray[np.float64], obj_id: int, color: str = 'grey', size: int = BLOCK_SIZE):
    self.state = state
    self.id = obj_id
    self.color = color
    self.size = size

class Square(Object):
  def __init__(self, state: npt.NDArray[np.float64], obj_id: int, color: str = 'grey', size: int = BLOCK_SIZE):
    super().__init__(state, obj_id, color, size)

  def draw(self, cells, canvas):
    x, y = self.state[0] * GRID_SIZE, self.state[1] * GRID_SIZE
    side = self.size/2.0
    cells[(x,y)] = [
      canvas.create_rectangle(x - side, y-side,
                                    x+side, y+side,
                                    fill=self.color, outline='black', width=2),
      canvas.create_text(x, y, text=self.id, fill=TEXT_COLOR),
    ]

class Arm(Square):
  def __init__(self, state: npt.NDArray[np.float64], obj_id: int, size: int = ARM_SIZE):
    super().__init__(state, obj_id, color, size)
    self.arm_empty = False

import sys

class ModelViewer:
  def __init__(self, width=500, height=250, side=GRID_SIZE, num_objects=4):

    self.width = width # window width
    self.height = height # window height
    self.side = side # size of an object 

    self.objects = []
    self.num_objects = num_objects

    # -----------  drawing related
    # ----------------------------

# -----------------------------------
# drawing
  def drawObjects(self, canvas, cells):
    """
    draw all the objects in the environment
    """
    for i in range(self.num_objects):
      self.objects[i].draw(cells, canvas)

  def clear(self):
    """
    clear the canvas
    """
    self.canvas.delete('all')


gcs = make_simple_transparent_gcs_test(2, 3, 15)

assert gcs.solution.is_success(), "Solution was not found"
modes, vertices = gcs.get_solution_path()
for i in range(len(vertices)):
    vertices[i] = ["%.1f" % v for v in vertices[i]]
mode_now = gcs.get_mode_from_vertex_name(modes[1])


for i in range(len(modes)):
    sg = vertices[i][0 : gcs.block_dim]
    if modes[i] == "start":
        INFO("Start at", sg)
    elif modes[i] == "target":
        INFO("Move to", sg, "; Finish")
    else:
        mode_next = gcs.get_mode_from_vertex_name(modes[i])
        if mode_next == mode_now:
            grasp = ""
        elif mode_next == 0:
            grasp = "Ungrasp block " + str(mode_now)
        else:
            grasp = "Grasp   block " + str(mode_next)
        mode_now = mode_next
        INFO("Move to", sg, "; " + grasp)