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
# from sim.util.utils import user_input
# import sys

import time

BLOCK_COLORS = ["#E3B5A4", "#E8D6CB", "#C3DFE0", "#F6E4F6", "#F4F4F4"]
ARM_COLOR = "#843B62"
ARM_NOT_EMPTY_COLOR = "#5E3886"  # 5E3886 621940

TEXT_COLOR = "#0B032D"
BLACK = "#0B032D"
BACKGROUND = "#F5E9E2"

ARM_SIZE = 20
BLOCK_SIZE = 25
GRID_SIZE = 30
OBSTACLE_COLOR = 'black'


class Draw2DSolution:
    def __init__(self, num_modes: int, ub: float, mode_solution, vertex_solution, goal):
        assert num_modes - 1 <= len(BLOCK_COLORS), "Not enough block colors"
        self.num_modes = num_modes
        self.ub = ub
        self.cell_scaling = 55
        self.block_size = 50
        self.arm_size = 50
        self.padding = self.block_size/2
        self.border = 10

        self.mode = mode_solution
        self.vertex = vertex_solution

        self.speed = 2  # units/s
        self.grasp_dt = 0.75  # s
        self.move_dt = 0.025  # s

        self.grasping = False

        self.width = self.cell_scaling * self.ub + self.padding * 2 + self.border * 2
        self.goal = goal

        self.tk = Tk()
        self.tk.withdraw()
        top = Toplevel(self.tk)
        top.wm_title("Moving Blocks")
        top.protocol("WM_DELETE_WINDOW", top.destroy)

        self.canvas = Canvas(
            top, width=self.width, height=self.width, background=BACKGROUND
        )
        self.canvas.pack()
        self.cells = {}
        self.environment = []

    def get_pixel_location(self, loc):
        return loc[0] * self.cell_scaling + self.padding + self.border, loc[1] * self.cell_scaling + self.padding + self.border

    def draw_solution(self):
        vertex = self.vertex
        mode = self.mode

        state_now = vertex[0, :]
        mode_now = mode[1]
        if mode_now == '0':
            self.grasping = False
        # draw initial state
        self.draw_state(state_now)
        time.sleep(1.0)

        for i in range(1, len(vertex)):
            state_next = vertex[i, :]
            mode_next = mode[i]

            self.move_from_to(state_now, state_next)
            if mode_now != mode_next:
                if mode_next == "target":
                    self.draw_state(state_next)
                    time.sleep(2.0)
                elif mode_next != '0':
                    self.grasping = True
                else:
                    self.grasping = False
                self.grasp(state_next)
            mode_now = mode_next
            state_now = state_next

    def move_from_to(self, state_now, state_next):
        delta = state_next - state_now
        distance = np.linalg.norm(delta[0:2])
        distance_per_dt = self.speed * self.move_dt
        num_steps = int(max(float(distance / distance_per_dt), 1.0))
        for i in range(1, num_steps + 1):
            self.draw_state(state_now + delta * i / num_steps)
            time.sleep(self.move_dt)

    def grasp(self, state):
        self.draw_state(state)
        time.sleep(self.grasp_dt)

    def draw_state(self, state):
        self.clear()
        self.draw_background()
        self.draw_goal()
        for i in range(1, self.num_modes):
            self.draw_block(state[2 * i : 2 * i + 2], i)

        self.draw_arm(state[0:2])
        self.tk.update()

    def clear(self):
        self.canvas.delete("all")

    def draw_block(self, block_state, block_num):
        x, y = self.get_pixel_location(block_state)
        side = self.block_size / 2.0
        self.cells[(x, y)] = [
            self.canvas.create_rectangle(
                x - side,
                y - side,
                x + side,
                y + side,
                fill=BLOCK_COLORS[block_num - 1],
                outline="black",
                width=2,
            ),
            self.canvas.create_text(x, y, text=block_num, fill=TEXT_COLOR),
        ]

    def draw_arm(self, arm_state):
        x, y = self.get_pixel_location(arm_state)
        side = self.arm_size / 2.0
        if self.grasping:
            arm_color = ARM_NOT_EMPTY_COLOR
        else:
            arm_color = ARM_COLOR

        self.cells[(x, y)] = [
            self.canvas.create_oval(
                x - side,
                y - side,
                x + side,
                y + side,
                fill=arm_color,
                outline="black",
                width=2,
            ),
            self.canvas.create_text(x, y, text="arm", fill=TEXT_COLOR),
        ]

    def draw_shadow(self, state, name):
        x, y = self.get_pixel_location(state)
        side = self.block_size / 2.0
        if name == 'arm':
            create_func = self.canvas.create_oval
        else:
            create_func = self.canvas.create_rectangle
        self.cells[(x, y)] = [
            create_func(
                x - side,
                y - side,
                x + side,
                y + side,
                fill='#D3D3D3',
                outline="grey",
                width=2,
            ),
            self.canvas.create_text(x, y, text=name, fill=TEXT_COLOR),
        ]

    def draw_background(self):
        num_lines = self.num_modes
        for i in range(1, num_lines+1):
            self.environment.append(self.canvas.create_line(0, i/(num_lines+1)*self.width, self.width, i/(num_lines+1)*self.width, fill='grey', width = 1 ))
            self.environment.append(self.canvas.create_line(i/(num_lines+1)*self.width, 0, i/(num_lines+1)*self.width, self.width, fill='grey', width = 1 ))
        self.environment.append([
            self.canvas.create_rectangle(0, 0, self.border, self.width, fill='black', outline='black', width=0),
            self.canvas.create_rectangle(0, 0, self.width, self.border, fill='black', outline='black', width=0),
            self.canvas.create_rectangle(0, self.width-self.border, self.width, self.width, fill='black', outline='black', width=0),
            self.canvas.create_rectangle(self.width-self.border, 0, self.width, self.width, fill='black', outline='black', width=0),
        ])

    def draw_goal(self):
        self.draw_shadow(self.goal[0:2], 'arm')
        for i in range(1, self.num_modes):
            self.draw_shadow(self.goal[2*i:2*i+2], i)
        




block_dim = 2
num_blocks = 3
horizon = 15

gcs, ub, goal = make_simple_transparent_gcs_test(block_dim, num_blocks, horizon)

assert gcs.solution.is_success(), "Solution was not found"
modes, vertices = gcs.get_solution_path()
for i in range(len(vertices)):
    vertices[i] = ["%.1f" % v for v in vertices[i]]

drawer = Draw2DSolution(num_blocks + 1, ub, modes, vertices, goal)
drawer.draw_solution()
