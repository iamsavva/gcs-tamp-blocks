# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

import pydrake.geometry.optimization as opt  # pylint: disable=import-error
from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    GraphOfConvexSets,
    HPolyhedron,
    ConvexSet,
)
from pydrake.solvers import (  # pylint: disable=import-error, unused-import
    Binding,
    L2NormCost,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearCost,
)
from pydrake.math import eq # pylint: disable=import-error


from .util import ERROR, WARN, INFO, YAY
from .gcs_options import GCSforBlocksOptions, EdgeOptions, EdgeOptExp
from .gcs import GCSforBlocks


class GCSforBlocksExp(GCSforBlocks):
    """
    GCS for N-dimensional block moving using a top-down suction cup.
    Specified for one in one out: every mode has a single node coming out that represents that mode
    and connected to the nodes of the next laer. This reduces the number of edges but makes the
    convex formulation more loose.
    """

    ###################################################################################
    # Adding layers of nodes (trellis diagram style)

    def build_the_graph(
        self,
        start_state: Point,
        start_mode: int,
        target_state: Point,
        target_mode: int,
    ) -> None:
        """
        Build the GCS graph of horizon H from start to target nodes.
        TODO:
        - allow target state to be a set
        """
        # reset the graph
        self.gcs = GraphOfConvexSets()
        self.graph_built = False
        self.start_mode = start_mode
        self.target_mode = target_mode

        # add all vertices
        self.add_all_vertices(start_state, target_state)
        # add all edges
        self.add_all_edges()

        self.graph_built = True

    def add_all_vertices(
        self,
        start_state: Point,
        target_state: Point,
    ) -> None:
        # add start node to the graph
        self.add_vertex(start_state, "start")

        free_set = self.set_gen.get_convex_set_experimental("free")
        grasp_set = self.set_gen.get_convex_set_experimental("grasping")
        
        # add each horizon_point
        for i in range(self.opt.horizon):
            # if self.opt.split_move:
            self.add_vertex(free_set, "F_" + str(i) )
            self.add_vertex(free_set, "FM_" + str(i) )
            self.add_vertex(grasp_set, "G_" + str(i) )
            self.add_vertex(grasp_set, "GM_" + str(i) )
            # else:
            #     self.add_vertex(free_set, "F_" + str(i) )
            #     self.add_vertex(grasp_set, "G_" + str(i) )
        self.add_vertex(free_set, "F_" + str(self.opt.horizon) )
        self.add_vertex(free_set, "FM_" + str(self.opt.horizon) )

        # add target vertex
        self.add_vertex(target_state, "target")



    def add_all_edges(self) -> None:
        # F -- FM -- G -- GM
        # add equality constraint between start at F_0
        self.add_edge("start", "F_0", 0, EdgeOptExp.equality_edge())
        self.add_edge("FM_0", "target", 0 , EdgeOptExp.equality_edge())

        for j in range(self.opt.horizon):
            i = str(j)
            i_1 = str(j+1)
            # add transition constraint F_ to FM_
            self.add_edge("F_"+i, "FM_"+i, 0, EdgeOptExp.move_edge())

            # add weird equality constraint FM_ to G_, per block
            for block in range(1, self.opt.num_blocks+1):
                self.add_edge("FM_"+i, "G_"+i, block, EdgeOptExp.grasp_edge())

            # add move constraint from G_ to GM_
            self.add_edge("G_"+i, "GM_"+i, 1, EdgeOptExp.move_edge())

            # add weird equality constraint from GM to F, per block
            for block in range(1, self.opt.num_blocks+1):
                self.add_edge("GM_"+i, "F_"+i_1, block, EdgeOptExp.ungrasp_edge())

            # target: add equalities with FM_
            self.add_edge("FM_"+i_1, "target", 0 , EdgeOptExp.equality_edge())
        self.add_edge("F_"+str(self.opt.horizon), "FM_"+str(self.opt.horizon), 0, EdgeOptExp.move_edge())


    def add_edge(
        self,
        left_name:str,
        right_name:str,
        block:int,
        edge_opt: EdgeOptExp,
    ) -> None:
        """
        READY
        Add an edge between two vertices, as well as corresponding constraints and costs.
        """
        # add an edge
        left_vertex = self.name_to_vertex[left_name]
        right_vertex = self.name_to_vertex[right_name]
        edge_name = "E"+str(block) + "_" + left_name +"-->"+right_name
        # self.get_edge_name(left_vertex.name(), right_vertex.name())
        edge = self.gcs.AddEdge(left_vertex, right_vertex, edge_name)

        # -----------------------------------------------------------------
        # Adding constraints
        # -----------------------------------------------------------------
        # add an orbital constraint
        if edge_opt.add_orbital_constraint:
            self.add_orbital_constraint_experimental(edge)
        if edge_opt.add_grasp_constraint:
            self.add_grasp_constraint(edge, block)
        if edge_opt.add_ungrasp_constraint:
            self.add_ungrasp_constraint(edge, block)
        if edge_opt.add_equality_constraint:
            self.add_point_equality_constraint(edge)
        # -----------------------------------------------------------------
        # Adding costs
        # -----------------------------------------------------------------
        # add movement cost on the edge
        if edge_opt.add_gripper_movement_cost:
            self.add_gripper_movement_cost_on_edge(edge)
        # add time cost on edge
        if edge_opt.add_grasp_cost:
            self.add_grasp_cost_on_edge(edge)


    ###################################################################################
    # Adding constraints and cost terms
    def add_orbital_constraint_experimental( self, edge: GraphOfConvexSets.Edge) -> None:
        xu, xv = edge.xu(), edge.xv()
        constraints = eq(xu[self.opt.block_dim:], xv[self.opt.block_dim:])
        for c in constraints:
            edge.AddConstraint(c)

    def add_grasp_constraint(self, edge, i):
        x, y = edge.xu(), edge.xv()
        b = self.opt.block_dim
        constraints = np.array([])
        constraints = np.append(constraints, eq(x[0:b], x[i*b:i*b+b]) )
        constraints = np.append(constraints, eq(x[0:b], y[0:b]) )
        for j in range(1, i):
            constraints = np.append(constraints, eq( x[j*b:j*b+b], y[j*b:j*b+b]) )
        for j in range(i+1, self.opt.num_blocks):
            constraints = np.append(constraints, eq( x[j*b+b:j*b+b+b], y[j*b:j*b+b]) )
        
        for j in range(self.opt.num_blocks):
            k = self.opt.block_dim*self.opt.num_blocks + j
            if j == i:
                constraints = np.append(constraints, eq(y[k], 1.0) )
            else:
                constraints = np.append(constraints, eq(y[k], 1.0) )
        for c in constraints:
            edge.AddConstraint(c)
        
    def add_ungrasp_constraint(self, edge, i):
        y, x = edge.xu(), edge.xv()
        b = self.opt.block_dim
        constraints = np.array([])
        constraints = np.append(constraints, eq(x[0:b], x[i*b:i*b+b]) )
        constraints = np.append(constraints, eq(x[0:b], y[0:b]) )
        for j in range(1, i):
            constraints = np.append(constraints, eq( x[j*b:j*b+b], y[j*b:j*b+b]) )
        for j in range(i+1, self.opt.num_blocks):
            constraints = np.append(constraints, eq( x[j*b+b:j*b+b+b], y[j*b:j*b+b]) )
        for j in range(self.opt.num_blocks):
            k = self.opt.block_dim*self.opt.num_blocks + j
            if j == i:
                constraints = np.append(constraints, eq(y[k], 1.0) )
            else:
                constraints = np.append(constraints, eq(y[k], 1.0) )
        for c in constraints:
            edge.AddConstraint(c)
        
    def add_point_equality_constraint(self, edge: GraphOfConvexSets.Edge) -> None:
        xu, xv = edge.xu(), edge.xv()
        constraints = eq(xu, xv)
        for c in constraints:
            edge.AddConstraint(c)

    def add_gripper_movement_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        L2 norm cost on the movement of the gripper.
        """
        xu, xv = edge.xu(), edge.xv()
        #  gripper state is 0 to block_dim
        d = self.opt.block_dim
        n = len(xu)
        A = np.zeros((d, 2 * n))
        A[:, 0:d] = np.eye(d)
        A[:, n : n + d] = -np.eye(d)
        b = np.zeros(d)
        # add the cost
        cost = L2NormCost(A, b)
        edge.AddCost(Binding[L2NormCost](cost, np.append(xv, xu)))

    def add_grasp_cost_on_edge(self, edge: GraphOfConvexSets.Edge) -> None:
        """
        READY
        Walking along the edges costs some cosntant term. This is done to avoid grasping and ungrasping in place.
        """
        n = len(edge.xv())
        a = np.zeros(n)
        b = self.opt.time_cost_weight * np.ones(1)
        cost = LinearCost(a, b)
        edge.AddCost(Binding[LinearCost](cost, edge.xv()))


    ###################################################################################
    # Solve and display solution

    def get_solution_path(self) -> T.Tuple[T.List[str], npt.NDArray]:
        """Given a solved GCS problem, and assuming it's tight, find a path from start to target"""
        # find edges with non-zero flow
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [self.solution.GetSolution(p) for p in flow_variables]
        active_edges = [
            edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow >= 0.99
        ]
        # using these edges, find the path from start to target
        path = self.find_path_to_target(active_edges, self.name_to_vertex["start"])
        modes = [v.name() for v in path]
        return modes

    def verbose_solution_description(self) -> None:
        """Describe the solution in text: grasp X, move to Y, ungrasp Z"""
        modes, vertices = self.get_solution_path()
        for i in range(len(vertices)):
            vertices[i] = ["%.1f" % v for v in vertices[i]]
        mode_now = modes[1]
        INFO("-----------------------")
        INFO("Solution is:")
        INFO("-----------------------")
        for i in range(len(modes)):  # pylint: disable=consider-using-enumerate
            sg = vertices[i][0 : self.opt.block_dim]
            if modes[i] == "start":
                INFO("Start at", sg)
            elif modes[i] == "target":
                INFO("Move to", sg, "; Finish")
            else:
                mode_next = modes[i]
                if mode_next == mode_now:
                    grasp = ""
                elif mode_next == "0":
                    grasp = "Ungrasp block " + str(mode_now)
                else:
                    grasp = "Grasp block " + str(mode_next)
                mode_now = mode_next
                INFO("Move to", sg, "; " + grasp)
