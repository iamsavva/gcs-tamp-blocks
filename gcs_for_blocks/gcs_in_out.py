# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

import pydrake.geometry.optimization as opt  # pylint: disable=import-error
from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
)

from .util import ERROR, WARN, INFO, YAY
from .gcs_options import GCSforBlocksOptions, EdgeOptions
from .gcs import GCSforBlocks


class GCSforBlocksOneInOneOut(GCSforBlocks):
    """
    GCS for N-dimensional block moving using a top-down suction cup.
    Specified for one in one out: every mode has a single node coming out that represents that mode
    and connected to the nodes of the next laer. This reduces the number of edges but makes the
    convex formulation more loose.
    """

    ###################################################################################
    # Adding layers of nodes (trellis diagram style)

    def add_all_vertices(
        self,
        start_state: Point,
        target_state: Point,
    ) -> None:
        # add start node to the graph
        self.add_vertex(start_state, "start")
        # add vertices into horizon 0
        for set_id in self.sets_per_mode[self.start_mode]:
            self.add_vertex(self.get_convex_set_for_set_id(set_id), self.get_vertex_name(0, set_id))
        # add vertices into horizon 1 through last
        for layer in range(1, self.opt.horizon):
            for mode in self.modes_per_layer[layer]:
                # for each set in that mode, add new vertex
                for set_id in self.sets_per_mode[mode]:
                    vertex_name = self.get_vertex_name(layer, set_id)
                    convex_set = self.get_convex_set_for_set_id(set_id)
                    self.add_vertex(convex_set, vertex_name)
        # add target vertex
        self.add_vertex(target_state, "target")

        # for each layer
        for layer in range(self.opt.horizon):
            # add in-out node for each mode
            for mode in self.modes_per_layer[layer]:
                vertex_name = self.get_vertex_name(layer, mode, True)
                convex_set = self.set_gen.get_convex_set_for_mode_polyhedron(mode)
                self.add_vertex(convex_set, vertex_name)

    def add_all_edges(self) -> None:
        ############################
        # between start and layer 0

        # get sets that intersect with the start set
        sets_with_start = self.get_sets_in_mode_that_intersect_with_set(
            self.start_mode,
            self.name_to_vertex["start"].set(),
            just_one=self.opt.connect_source_target_to_single_set,
        )
        names_of_sets_with_start = self.set_names_for_layer(sets_with_start, 0)
        self.connect_to_vertex_on_the_right(
            "start", names_of_sets_with_start, EdgeOptions.equality_edge()
        )

        ############################
        # edges within layers
        for layer in range(self.opt.horizon):
            for mode in self.modes_per_layer[layer]:
                # add edges within modes
                for set_id in self.sets_per_mode[mode]:
                    vertex_name = self.get_vertex_name(layer, set_id)
                    # edges out of vertex into the same mode of same layer
                    intra_mode_in = self.get_edges_within_same_mode(set_id)
                    names_of_intra_mode = self.set_names_for_layer(intra_mode_in, layer)
                    self.connect_to_vertex_on_the_right(
                        vertex_name, names_of_intra_mode, EdgeOptions.within_mode_edge()
                    )
                # add edges into next mode-out points
                in_out_name = self.get_vertex_name(layer, mode, True)
                left_vertex_names = self.set_names_for_layer(self.sets_per_mode[mode], layer)
                self.connect_to_vertex_on_the_left(
                    left_vertex_names, in_out_name, EdgeOptions.into_in_out_edge()
                )
                # add edges from in-out into next layer:
                if layer < self.opt.horizon - 1:
                    # for each mode in the next layer
                    # technically this connectivity needs to be handled more carefully
                    for right_mode in self.modes_per_layer[layer + 1]:
                        right_vertex_names = self.set_names_for_layer(
                            self.sets_per_mode[right_mode], layer + 1
                        )
                        self.connect_to_vertex_on_the_right(
                            in_out_name,
                            right_vertex_names,
                            EdgeOptions.out_of_in_out_edge(),
                        )

        ##############################
        # edges to target
        # sets in target_mode that intersect with target_state

        names_of_sets_with_target = []
        # at each horizon level, only sets that contain the target can transition into target
        for layer in range(self.opt.horizon):
            # if that layer has a target mode
            if self.target_mode in self.modes_per_layer[layer]:
                names_of_sets_with_target += [self.get_vertex_name(layer, self.target_mode, True)]
        # add the edges
        self.connect_to_vertex_on_the_left(
            names_of_sets_with_target, "target", EdgeOptions.equality_edge()
        )

    ###################################################################################
    # Vertex and edge naming

    def get_vertex_name(self, layer: int, set_id: int, transition: bool = False) -> str:
        """
        Naming convention is:
            M_<layer>_<set_id> for regular nodes
            M_<layer>_T_<mode> for in-out node that goes into the layer
        """
        if transition:
            return "M_" + str(layer) + "_T_" + str(set_id)
        return "M_" + str(layer) + "_" + str(set_id)

    def get_set_id_from_vertex_name(self, name: str) -> int:
        assert name not in ("start", "target"), "Trying to get set id for bad sets!"
        assert len(name.split("_")) == 3, "Trying to get set id for an in-out set!"
        set_id = int(name.split("_")[-1])
        return set_id

    def get_mode_from_vertex_name(self, name: str) -> int:
        if len(name.split("_")) == 4:
            return int(name.split("_")[-1])
        if name == "start":
            return self.start_mode
        if name == "target":
            return self.target_mode
        set_id = self.get_set_id_from_vertex_name(name)
        return self.get_mode_from_set_id(set_id)

    ###################################################################################
    # Solve and display solution

    def get_solution_path(self) -> T.Tuple[T.List[str], npt.NDArray]:
        """Given a solved GCS problem, and assuming it's tight, find a path from start to target"""
        assert self.graph_built, "Must build graph first!"
        assert self.solution.is_success(), "Solution was not found"
        # find edges with non-zero flow
        flow_variables = [e.phi() for e in self.gcs.Edges()]
        flow_results = [self.solution.GetSolution(p) for p in flow_variables]
        active_edges = [edge for edge, flow in zip(self.gcs.Edges(), flow_results) if flow >= 0.99]
        # using these edges, find the path from start to target
        path = self.find_path_to_target(active_edges, self.name_to_vertex["start"])
        modes = [v.name() for v in path]
        modes = [
            str(self.get_mode_from_vertex_name(mode)) if mode not in ("start", "target") else mode
            for mode in modes
        ]
        vertex_values = np.vstack([self.solution.GetSolution(v.x()) for v in path])
        return modes, vertex_values

    def verbose_solution_description(self) -> None:
        """Describe the solution in text: grasp X, move to Y, ungrasp Z"""
        assert self.solution.is_success(), "Solution was not found"
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
