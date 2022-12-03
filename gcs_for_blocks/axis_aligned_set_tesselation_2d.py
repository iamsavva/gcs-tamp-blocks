import typing as T
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.pyplot import cm

class AlignedSet:
    # axis aligned set
    def __init__(self, a, b, l, r, name = ""):
        # above bound a
        # below bound b
        # left  bound l
        # right bound r
        self.constraints = {"a": a, "b":b, "l":l, "r":r} # type: T.Dict[str, float]
        self.name = name # type: str

    @property
    def l(self):
        return self.constraints["l"]
    @property
    def r(self):
        return self.constraints["r"] 
    @property
    def a(self):
        return self.constraints["a"]
    @property
    def b(self):
        return self.constraints["b"]
    @property
    def set_is_non_empty(self):
        if self.l >= self.r or self.a <= self.b:
            return False
        return True
    @property
    def set_is_obstacle(self):
        return self.name != ""


    def new_constraint_is_stronger(self, dir:str, new_val:float):
        if dir in ("b", "l") and self.constraints[dir] < new_val:
            return True
        elif dir in ("a", "r") and self.constraints[dir] > new_val:
            return True
        return False

    def add_constraints(self, constraints):
        # return true if set is non empty after adding constraints
        for (dir, val) in constraints:
            if self.new_constraint_is_stronger(dir, val):
                self.constraints[dir] = val
        return self.set_is_non_empty

    def intersects_with(self, other):
        if self.r <= other.l or other.r <= self.l or self.a <= other.b or other.a <= self.b:
            return False
        return True

    def share_edge(self, other):
        a = min(self.a, other.a)
        b = max(self.b, other.b)
        l = max(self.l, other.l)
        r = min(self.r, other.r)
        if ((a-b) > 0 and np.isclose(l,r)) or ((r-l) > 0 and np.isclose(b,a)):
            return True
        return False

    def intersection(self, other:"AlignedSet"):
        assert self.intersects_with(other), "sets don't intersect"
        a = min(self.a, other.a)
        b = max(self.b, other.b)
        l = max(self.l, other.l)
        r = min(self.r, other.r)
        return AlignedSet(a=a,b=b,l=l,r=r)

    def is_inside(self, box):
        if self.l >= box.l and self.r <= box.r and self.b >= box.b and self.a <= box.a:
            return True
        return False
    
    def __repr__(self):
        return "L" + str(self.l) + " R" + str(self.r) + " B" + str(self.b) + " A" + str(self.a)

    def get_direction_sets(self, bounding_box: "AlignedSet"):
        assert self.is_inside(bounding_box)
        sets = []
        # left
        sets.append( AlignedSet(l = bounding_box.l, r = self.l, a = self.a, b = self.b ) )
        # right
        sets.append( AlignedSet(r = bounding_box.r, l = self.r, a = self.a, b = self.b ) )
        # below
        sets.append( AlignedSet(r = bounding_box.r, l = bounding_box.l, a = self.b, b = bounding_box.b ) )
        # above
        sets.append( AlignedSet(r = bounding_box.r, l = bounding_box.l, b = self.a, a = bounding_box.a ) )
        return sets

    def get_rectangle(self, color):    
        return patches.Rectangle((self.l, self.b), self.r-self.l, self.a-self.b, linewidth=1, edgecolor='black', facecolor=color, label=self.name)


def axis_aligned_tesselation(bounding_box: AlignedSet, obstacles:T.List[AlignedSet]):
    all_sets = set() # type: T.Set[AlignedSet]
    all_sets.add(bounding_box)
    indd = 0
    temp_box_index = 0
    for obstacle in obstacles:
        new_sets = []
        rem_sets = []
        # add the obstacle
        for box in all_sets:
            if obstacle.intersects_with(box):
                # TODO: ultimate goal is to drop this, right
                assert not box.set_is_obstacle, "Shouldn't have intersecting boxes"
                # if obstacle intersects with some box
                rem_sets.append(box)
                direction_sets_for_obstacle = obstacle.get_direction_sets(bounding_box)
                # add all LRAB intersections
                for dir_set in direction_sets_for_obstacle:
                    if box.intersects_with(dir_set):
                        new_sets.append( box.intersection(dir_set))
                        temp_box_index += 1

        new_sets.append(obstacle)
        for add_me in new_sets:
            all_sets.add(add_me)
        for rem in rem_sets:
            all_sets.remove(rem)
        indd += 1
    return list(all_sets)

def locations_to_aligned_sets(start, target, block_width):
    bw = block_width
    sets = []
    for i, (x,y) in enumerate(start):
        sets.append( AlignedSet(l=x-bw, r=x+bw, b=y-bw, a=y+bw, name="s"+str(i)))
    for i, (x,y) in enumerate(target):
        sets.append( AlignedSet(l=x-bw, r=x+bw, b=y-bw, a=y+bw, name="t"+str(i)))
    return sets

def plot_list_of_aligned_sets(sets, bounding_box):
    fig, ax = plt.subplots()
    index = 0
    print(len(sets))
    for a_set in sets:
        index+=1
        ax.add_patch(a_set.get_rectangle(colors[index]))
        if a_set.set_is_obstacle:
            ax.annotate(a_set.name, ((a_set.l+a_set.r)/2, (a_set.b+a_set.a)/2), color='black', weight='bold', fontsize=10, ha='center', va='center')

    ax.set_xlim([bounding_box.l,bounding_box.r])
    ax.set_ylim([bounding_box.b, bounding_box.a])
    plt.show()

def get_edges(sets):
    for i in range(len(sets)):
        for j in range(i, len(sets)):
            if sets[i].share_edge( sets[j] ):
                print( sets[i].name, sets[j].name )

colors = cm.rainbow(np.linspace(0, 1, 30))

bounding_box = AlignedSet(b=0,a=12,l=0,r=12)
block_width = 1
# start = [(1,1), (3,5), (7,4)]
# target = [(5,11), (9,7), (5,8)]
start = [(1,1)]
target = [(5,11)]

obstacles = locations_to_aligned_sets(start,target, block_width)
sets = axis_aligned_tesselation(bounding_box, obstacles)
# index the boxes
index = 0
for s in sets:
    if not s.set_is_obstacle:
        s.name = "r" + str(index)
        index += 1

get_edges(sets)
plot_list_of_aligned_sets(sets, bounding_box)








                






