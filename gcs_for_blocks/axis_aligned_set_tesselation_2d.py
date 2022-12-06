import typing as T

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Box:
    """
    Simple class for defining axis aligned boxes and getting their half space representations.
    """

    def __init__(self, lb: npt.NDArray, ub: npt.NDArray, state_dim: int):
        assert state_dim == len(lb)
        assert state_dim == len(ub)
        self.lb = lb
        self.ub = ub
        self.state_dim = state_dim

    def get_hpolyhedron(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """Returns an hpolyhedron for the box"""
        # Ax <= b
        A = np.vstack((np.eye(self.state_dim), -np.eye(self.state_dim)))
        b = np.hstack((self.ub, -self.lb))
        return A, b

    def get_perspective_hpolyhedron(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """Returns a perspective hpolyhedron for the box"""
        # Ax <= b * lambda
        # Ax - b * lambda <= 0
        # [A -b] [x, lambda]^T <= 0
        A, b = self.get_hpolyhedron()
        b.resize((2 * self.state_dim, 1))
        pA = np.hstack((A, (-1) * b))
        pb = np.zeros(2 * self.state_dim)
        return pA, pb


class AlignedSet:
    """
    A class that defines a 2D axis aligned set and relevant tools.
    """

    def __init__(
        self,
        a: float,
        b: float,
        l: float,
        r: float,
        name: str = "",
        obstacles: T.List[T.Tuple[str, int]] = [],
    ) -> None:
        # above bound a, y <= a
        # below bound b, b <= y
        # left  bound l, l <= x
        # right bound r, x <= r
        self.constraints = {"a": a, "b": b, "l": l, "r": r}  # type: T.Dict[str, float]
        self.name = name  # type: str
        self.box = Box(lb=np.array([l, b]), ub=np.array([r, a]), state_dim=2)  # type: Box
        self.obstacles = obstacles

    @property
    def l(self) -> float:
        return self.constraints["l"]

    @property
    def r(self) -> float:
        return self.constraints["r"]

    @property
    def a(self) -> float:
        return self.constraints["a"]

    @property
    def b(self) -> float:
        return self.constraints["b"]

    def offset_in(self, delta):
        self.constraints["l"] = self.l + delta
        self.constraints["r"] = self.r - delta
        self.constraints["b"] = self.b + delta
        self.constraints["a"] = self.a - delta

    def copy(self) -> "AlignedSet":
        return AlignedSet(
            a=self.a, b=self.b, l=self.l, r=self.r, name=self.name, obstacles=self.obstacles
        )

    def intersects_with(self, other) -> bool:
        """
        Instead of working with tight bounds, offset all boxes inwards by a small amount.
        Interseciton = fully right of or left of or above or below
        """
        # intersection by edge is fine
        return not (
            self.r <= other.l or other.r <= self.l or self.a <= other.b or other.a <= self.b
        )

    def point_is_in_set(self, point: T.List[float]):
        """True if point is inside a set"""
        return self.l <= point[0] <= self.r and self.b <= point[1] <= self.a

    def share_edge(self, other: "AlignedSet", rtol=0.000001) -> bool:
        """
        Two sets share an edge if they intersect
            + left of one is right of another  or  below of one is above of another.
        """
        b, a = max(self.b, other.b), min(self.a, other.a)
        l, r = max(self.l, other.l), min(self.r, other.r)
        return ((a - b) > 0 and np.isclose(l, r, rtol)) or ((r - l) > 0 and np.isclose(b, a, rtol))

    def intersection(self, other: "AlignedSet"):
        """Intersection of two sets; cannot be just an edge (i.e., without interior)"""
        assert self.intersects_with(other), "sets don't intersect"
        b, a = max(self.b, other.b), min(self.a, other.a)
        l, r = max(self.l, other.l), min(self.r, other.r)
        return AlignedSet(a=a, b=b, l=l, r=r)

    def is_inside(self, box: "AlignedSet"):
        return self.l >= box.l and self.r <= box.r and self.b >= box.b and self.a <= box.a

    def __repr__(self):
        return "L" + str(self.l) + " R" + str(self.r) + " B" + str(self.b) + " A" + str(self.a)

    def get_direction_sets(self, bounding_box: "AlignedSet"):
        """A box tesselates a space into 5 sets: above / below / left / right / itself"""
        assert self.is_inside(bounding_box)
        dir_sets = []
        # left
        dir_sets.append(AlignedSet(l=bounding_box.l, r=self.l, a=self.a, b=self.b))
        # right
        dir_sets.append(AlignedSet(r=bounding_box.r, l=self.r, a=self.a, b=self.b))
        # below
        dir_sets.append(AlignedSet(r=bounding_box.r, l=bounding_box.l, a=self.b, b=bounding_box.b))
        # above
        dir_sets.append(AlignedSet(r=bounding_box.r, l=bounding_box.l, b=self.a, a=bounding_box.a))
        # itself
        dir_sets.append(self.copy())
        return dir_sets

    def get_rectangle(self, color: str):
        return patches.Rectangle(
            (self.l, self.b),
            self.r - self.l,
            self.a - self.b,
            linewidth=1,
            edgecolor="black",
            facecolor=color,
            label=self.name,
        )

    def get_hpolyhedron(self):
        # Ax <= b
        return self.box.get_hpolyhedron()

    def get_perspective_hpolyhedron(self):
        # Ax <= b phi
        return self.box.get_perspective_hpolyhedron()


def axis_aligned_tesselation(bounding_box: AlignedSet, obstacles: T.List[AlignedSet]):
    """
    Given a set of obstacles inside a bounding box, tesselate the space into axis-aligned boxes
    """
    # initialize the tesselation with the bounding box
    tesselation = set()  # type: T.Set[AlignedSet]
    tesselation.add(bounding_box)
    # for each obstacle that i need to add
    for obstacle in obstacles:
        new_sets, rem_sets = [], []
        # for each box that's already in the tesselation
        for box in tesselation:
            # if obstacle intersects with some box
            if obstacle.intersects_with(box):
                # remove that box
                rem_sets.append(box)
                # get 5 direction sets for the obstacle
                direction_sets_for_obstacle = obstacle.get_direction_sets(bounding_box)
                # add their intersections
                # TODO: add here the bit no-iterior intersections
                for dir_set in direction_sets_for_obstacle:
                    if box.intersects_with(dir_set):
                        intersection_set = box.intersection(dir_set)
                        intersection_set.obstacles = box.obstacles + dir_set.obstacles
                        new_sets.append(intersection_set)
        for add_me in new_sets:
            tesselation.add(add_me)
        for rem in rem_sets:
            tesselation.remove(rem)
    tesselation = list(tesselation)
    # name all the boxes
    tesselation_dict = dict()
    index = 0
    for s in tesselation:
        s.name, index = "r" + str(index), index + 1
        tesselation_dict[s.name] = s

    # check yourself: assert that no sets intersect
    for i, x in enumerate(tesselation):
        for j, y in enumerate(tesselation):
            if i < j:
                assert not x.intersects_with(y), "\n" + x.__repr__() + "\n" + y.__repr__()
    return tesselation_dict


def get_obstacle_to_set_mapping(
    start_block_pos: T.List[npt.NDArray],
    target_block_pos: T.List[npt.NDArray],
    convex_set_tesselation: T.Dict[str, AlignedSet],
) -> T.Dict[str, str]:
    """
    Block-specific function; per obstacle: find a set in which it is located.
    """
    obstacle_to_set = dict()
    for (i, pos) in enumerate(start_block_pos):
        for aset in convex_set_tesselation.values():
            # if point is in the set -- that's it
            if aset.point_is_in_set(pos):
                obstacle_to_set["s" + str(i)] = aset.name
                break

    for (i, pos) in enumerate(target_block_pos):
        for aset in convex_set_tesselation.values():
            # if point is in the set -- that's it
            if aset.point_is_in_set(pos):
                obstacle_to_set["t" + str(i)] = aset.name
                break
    return obstacle_to_set


def locations_to_aligned_sets(
    start: T.List[T.Tuple[float]],
    target: T.List[T.Tuple[float]],
    block_width: float,
    bounding_box: AlignedSet,
):
    """"""
    # note that "obstacle set" is double the width -- it's all the points where another box can't be
    bw = block_width
    obstacle_sets = []
    for i, (x, y) in enumerate(start):
        obst = AlignedSet(l=x - bw, r=x + bw, b=y - bw, a=y + bw, name="s" + str(i))
        nobst = obst.intersection(bounding_box.copy())
        nobst.name = "s" + str(i)
        nobst.obstacles = [("s", i)]
        obstacle_sets.append(nobst)
    for i, (x, y) in enumerate(target):
        obst = AlignedSet(l=x - bw, r=x + bw, b=y - bw, a=y + bw, name="t" + str(i))
        nobst = obst.intersection(bounding_box.copy())
        nobst.name = "t" + str(i)
        nobst.obstacles = [("t", i)]
        obstacle_sets.append(nobst)
    return obstacle_sets


def plot_list_of_aligned_sets(
    obstacles: T.List[T.Tuple[float]],
    tesselation: T.Dict[str, AlignedSet],
    bounding_box: AlignedSet,
    block_width: float,
):
    _, ax = plt.subplots()
    # for each set
    for a_set in tesselation.values():
        # if it's an obstacle -- make it grey, else -- white
        if len(a_set.obstacles) > 0:
            print(a_set.name, a_set.obstacles)
            color = "grey"
        else:
            color = "white"
        ax.add_patch(a_set.get_rectangle(color))
        # add a name and a set of obstacles that it represents
        ax.annotate(
            a_set.name + "\n" + str(a_set.obstacles),
            ((a_set.l + a_set.r) / 2, (a_set.b + a_set.a) / 2),
            color="black",
            weight="bold",
            fontsize=8,
            ha="center",
            va="center",
        )
    # center of obstacles
    ax.scatter([x for (x, y) in obstacles], [y for (x, y) in obstacles], color="mediumblue")
    # draw the blocks
    for (x, y) in obstacles:
        ax.add_patch(
            patches.Rectangle(
                (x - block_width / 2, y - block_width / 2),
                block_width,
                block_width,
                linewidth=1,
                edgecolor="white",
                facecolor="mediumblue",
                alpha=0.3,
            )
        )
    # axis stuff
    ax.set_xlim([bounding_box.l, bounding_box.r])
    ax.set_ylim([bounding_box.b, bounding_box.a])
    ax.axis("equal")
    plt.show()


if __name__ == "__main__":
    small_delta = 0.00001
    block_width = 1
    block_width_minus_delta = block_width - small_delta
    half_block_width = block_width / 2
    half_block_width_minus_delta = block_width_minus_delta / 2

    # MUST offset the bounding box
    bounding_box = AlignedSet(b=0, a=3, l=0, r=7)

    start = [(1 - 0.5, 1 - 0.5), (1 - 0.5, 2 - 0.5), (3 - 0.5, 1 - 0.5)]
    target = [(7 - 0.5, 1 - 0.5), (7 - 0.5, 3 - 0.5), (5 - 0.5, 1 - 0.5)]

    bounding_box = AlignedSet(b=0, a=6, l=0, r=7)
    start = [
        (1 - 0.5, 5 - 0.5),
        (1 - 0.5, 1 - 0.5),
        (1 - 0.5, 3 - 0.5),
        (3 - 0.5, 3 - 0.5),
        (3 - 0.5, 1 - 0.5),
        (1 - 0.5, 5 - 0.5),
        (3 - 0.5, 5 - 0.5),
    ]
    target = [
        (1 - 0.5, 6 - 0.5),
        (7 - 0.5, 1 - 0.5),
        (5 - 0.5, 1 - 0.5),
        (5 - 0.5, 3 - 0.5),
        (5 - 0.5, 5 - 0.5),
        (7 - 0.5, 5 - 0.5),
        (7 - 0.5, 3 - 0.5),
    ]
    fast = True

    bounding_box = AlignedSet(b=0, a=3, l=0, r=7)
    start = [
        (4 - 0.5, 2 - 0.5),
        (1 - 0.5, 1 - 0.5),
        (1 - 0.5, 3 - 0.5),
        (3 - 0.5, 3 - 0.5),
        (3 - 0.5, 1 - 0.5),
    ]
    target = [
        (4 - 0.5, 2 - 0.5),
        (7 - 0.5, 1 - 0.5),
        (7 - 0.5, 3 - 0.5),
        (5 - 0.5, 3 - 0.5),
        (5 - 0.5, 1 - 0.5),
    ]

    bounding_box = AlignedSet(b=0, a=2, l=0, r=6)
    start = [(1 - 0.5, 1 - 0.5), (1 - 0.5, 2 - 0.5), (2 - 0.5, 2 - 0.5), (2 - 0.5, 1 - 0.5)]
    target = [(6 - 0.5, 1 - 0.5), (6 - 0.5, 2 - 0.5), (5 - 0.5, 2 - 0.5), (5 - 0.5, 1 - 0.5)]

    convex_relaxation = False

    bounding_box.offset_in(half_block_width_minus_delta)
    obstacles = locations_to_aligned_sets(start, target, block_width_minus_delta, bounding_box)
    tesselation = axis_aligned_tesselation(bounding_box, obstacles)

    plot_list_of_aligned_sets(start + target, tesselation, bounding_box, block_width_minus_delta)
