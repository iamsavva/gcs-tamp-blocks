import typing as T

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm


class Box:
    """
    Simple class for defining axis aligned boxes and getting their half space representations.
    """
    def __init__(self, lb:npt.NDArray, ub:npt.NDArray, state_dim:int):
        assert state_dim == len(lb)
        assert state_dim == len(ub)
        self.lb = lb
        self.ub = ub
        self.state_dim = state_dim

    def get_hpolyhedron(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """ Returns an hpolyhedron for the box"""
        # Ax <= b 
        A = np.vstack((np.eye(self.state_dim), -np.eye(self.state_dim)))
        b = np.hstack((self.ub, -self.lb))
        return A, b

    def get_perspective_hpolyhedron(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        """ Returns a perspective hpolyhedron for the box"""
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
    def __init__(self, a:float, b:float, l:float, r:float, name:str="", obstacles:T.List[T.Tuple[str,int]]=[])->None:
        # above bound a, y <= a
        # below bound b, b <= y
        # left  bound l, l <= x
        # right bound r, x <= r
        self.constraints = {"a": a, "b": b, "l": l, "r": r}  # type: T.Dict[str, float]
        self.name = name  # type: str
        self.box = Box(lb=np.array([l, b]), ub=np.array([r, a]), state_dim=2) # type: Box
        self.obstacles = obstacles

    @property
    def l(self)->float:
        return self.constraints["l"]

    @property
    def r(self)->float:
        return self.constraints["r"]

    @property
    def a(self)->float:
        return self.constraints["a"]

    @property
    def b(self)->float:
        return self.constraints["b"]

    @property
    def set_is_non_empty(self)->bool:
        # TODO: this should be strict
        if self.l >= self.r or self.a <= self.b:
            return False
        return True

    # @property
    # def set_is_non_empty(self):
    #     if self.l >= self.r or self.a <= self.b:
    #         return False
    #     return True

    @property
    def set_is_obstacle(self)->bool: 
        # TODO: remove this altogether? check for intersecting boxes elsewhere
        return self.name != ""

    def copy(self)->"AlignedSet":
        return AlignedSet(a=self.a, b=self.b, l=self.l, r=self.r, name=self.name, obstacles=self.obstacles)

    def intersects_with(self, other)->bool:
        # TODO: again,equality constraints man
        # strictly right of, strictly left of, strictly above, strictly below
        if self.r <= other.l or other.r <= self.l or self.a <= other.b or other.a <= self.b:
            # if self.r < other.l or other.r < self.l or self.a < other.b or other.a < self.b:
            return False
        return True

    def share_edge(self, other:"AlignedSet")->bool:
        """
        Two sets share an edge if they intersect
            + left of one is right of another  or  below of one is above of another.
        """
        b, a = max(self.b, other.b), min(self.a, other.a)
        l, r = max(self.l, other.l), min(self.r, other.r)
        if ((a - b) > 0 and np.isclose(l, r)) or ((r - l) > 0 and np.isclose(b, a)):
            return True
        return False

    def intersection(self, other: "AlignedSet"):
        # TODO: intersection of two sets can be an edge?
        # boxes can intersect too
        assert self.intersects_with(other), "sets don't intersect"
        b, a = max(self.b, other.b), min(self.a, other.a)
        l, r = max(self.l, other.l), min(self.r, other.r)
        return AlignedSet(a=a, b=b, l=l, r=r)

    def is_inside(self, box):
        # NOTE: this is good to go and accurate
        if self.l >= box.l and self.r <= box.r and self.b >= box.b and self.a <= box.a:
            return True
        return False

    def __repr__(self):
        return "L" + str(self.l) + " R" + str(self.r) + " B" + str(self.b) + " A" + str(self.a)

    def get_direction_sets(self, bounding_box: "AlignedSet"):
        # NOTE: this is good to go
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
    all_sets = set()  # type: T.Set[AlignedSet]
    all_sets.add(bounding_box)
    temp_box_index = 0
    for obstacle in obstacles:
        new_sets = []
        rem_sets = []
        # add the obstacle
        for box in all_sets:
            if obstacle.intersects_with(box):
                assert not box.set_is_obstacle, "Shouldn't have intersecting boxes"
                # if obstacle intersects with some box
                rem_sets.append(box)
                direction_sets_for_obstacle = obstacle.get_direction_sets(bounding_box)
                # add all LRAB intersections
                for dir_set in direction_sets_for_obstacle:
                    if box.intersects_with(dir_set):
                        new_sets.append(box.intersection(dir_set))
                        temp_box_index += 1

        new_sets.append(obstacle)
        for add_me in new_sets:
            all_sets.add(add_me)
        for rem in rem_sets:
            all_sets.remove(rem)

    all_sets = list(all_sets)

    # index the boxes
    all_sets_dict = dict()
    index = 0
    for s in all_sets:
        if not s.set_is_obstacle:
            s.name = "r" + str(index)
            index += 1
        all_sets_dict[s.name] = s

    # assert that no sets intersect
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            assert not all_sets[i].intersects_with(all_sets[j]), (
                "\n" + all_sets[i].__repr__() + "\n" + all_sets[j].__repr__()
            )
    return all_sets_dict


def locations_to_aligned_sets(start, target, block_width, bounding_box):
    bw = block_width
    sets = []
    for i, (x, y) in enumerate(start):
        obst = AlignedSet(l=x - bw, r=x + bw, b=y - bw, a=y + bw, name="s" + str(i))
        nobst = obst.intersection(bounding_box)
        nobst.name = "s" + str(i)
        nobst.obstacles = [("s", i)]
        sets.append(nobst)
    for i, (x, y) in enumerate(target):
        obst = AlignedSet(l=x - bw, r=x + bw, b=y - bw, a=y + bw, name="t" + str(i))
        nobst = obst.intersection(bounding_box)
        nobst.name = "t" + str(i)
        nobst.obstacles = [("t", i)]
        sets.append(nobst)
        # sets.append(AlignedSet(l=x - bw, r=x + bw, b=y - bw, a=y + bw, name="t" + str(i)))
    return sets


def plot_list_of_aligned_sets(
    sets, bounding_box, visitations=None, moving_block_index=None, loc_path=None
):
    colors = cm.rainbow(np.linspace(0, 1, 30))
    _, ax = plt.subplots()
    index = 0
    for a_set in sets.values():
        index += 1
        if visitations is None:
            color = colors[index]
        else:
            if a_set.name[0] == "s" and visitations[int(a_set.name[1:])] == 0:
                color = "grey"
            elif a_set.name[0] == "t" and visitations[int(a_set.name[1:])] == 1:
                color = "grey"
            else:
                color = "white"
            if a_set.name[0] in ("s", "t") and int(a_set.name[1:]) == moving_block_index:
                color = "limegreen"

        ax.add_patch(a_set.get_rectangle(color))
        if a_set.set_is_obstacle:
            ax.annotate(
                a_set.name,
                ((a_set.l + a_set.r) / 2, (a_set.b + a_set.a) / 2),
                color="black",
                weight="bold",
                fontsize=10,
                ha="center",
                va="center",
            )
    if loc_path is not None:
        ax.plot(
            [x[0] for x in loc_path],
            [x[1] for x in loc_path],
            color="mediumblue",
            linewidth=3,
        )

    ax.set_xlim([bounding_box.l, bounding_box.r])
    ax.set_ylim([bounding_box.b, bounding_box.a])
    plt.show()


if __name__ == "__main__":
    bounding_box = AlignedSet(b=0, a=12, l=0, r=12)
    block_width = 1
    start = [(1, 1), (3, 5), (7, 4)]
    target = [(5, 11), (9, 7), (5, 8)]
    # start = [(1,1)]
    # target = [(5,11)]

    obstacles = locations_to_aligned_sets(start, target, block_width, bounding_box)
    sets = axis_aligned_tesselation(bounding_box, obstacles)

    # fix
    colors = cm.rainbow(np.linspace(0, 1, 30))
    plot_list_of_aligned_sets(sets, bounding_box)
