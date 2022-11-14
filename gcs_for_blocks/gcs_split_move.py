# pyright: reportMissingImports=false
import typing as T

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
)

from .gcs_options import EdgeOptions
from .gcs import GCSforBlocks


class GCSforBlocksSplitMove(GCSforBlocks):
    """
    GCS for N-dimensional block moving using a top-down suction cup.
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
            self.add_vertex(
                self.get_convex_set_for_set_id(set_id), self.get_vertex_name(0, set_id)
            )
            self.add_vertex(
                self.get_convex_set_for_set_id(set_id),
                self.get_vertex_name(0, set_id, "T"),
            )
        # add vertices into horizon 1 through last
        for layer in range(1, self.opt.horizon):
            for mode in self.modes_per_layer[layer]:
                # for each set in that mode, add new vertex
                for set_id in self.sets_per_mode[mode]:
                    vertex_name = self.get_vertex_name(layer, set_id)
                    convex_set = self.get_convex_set_for_set_id(set_id)
                    self.add_vertex(convex_set, vertex_name)
                    if mode == 0:
                        self.add_vertex(convex_set, "T" + vertex_name[1:])
        # add target vertex
        self.add_vertex(target_state, "target")

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
        self.connect_to_vertex_on_the_left(
            names_of_sets_with_start,
            "T" + names_of_sets_with_start[0][1:],
            EdgeOptions.within_mode_edge(),
        )

        ############################
        # edges within layers and between layers
        for layer in range(self.opt.horizon):
            for mode in self.modes_per_layer[layer]:
                # now add edges
                # TODO: this should be done more carefully in the future (?)
                for set_id in self.sets_per_mode[mode]:
                    vertex_name = self.get_vertex_name(layer, set_id)

                    # add edges into vertex from the previous layer
                    if layer > 0:
                        edges_in = self.get_edges_into_set_out_of_mode(set_id)
                        names_of_edges_in = self.set_names_for_layer(
                            edges_in, layer - 1
                        )
                        if mode != 0:
                            names_of_edges_in = ["T" + x[1:] for x in names_of_edges_in]
                            self.connect_to_vertex_on_the_left(
                                names_of_edges_in,
                                vertex_name,
                                EdgeOptions.mode_transition_edge(
                                    self.opt.add_grasp_cost
                                ),
                            )
                        else:
                            self.connect_to_vertex_on_the_left(
                                names_of_edges_in,
                                vertex_name,
                                EdgeOptions.between_modes_edge(self.opt.add_grasp_cost),
                            )
                            self.connect_to_vertex_on_the_left(
                                [vertex_name],
                                "T" + vertex_name[1:],
                                EdgeOptions.within_mode_edge(),
                            )

                    # edges out of vertex into the same mode of same layer
                    intra_mode_in = self.get_edges_within_same_mode(set_id)
                    names_of_intra_mode = self.set_names_for_layer(intra_mode_in, layer)
                    self.connect_to_vertex_on_the_right(
                        vertex_name, names_of_intra_mode, EdgeOptions.within_mode_edge()
                    )

        ##############################
        # edges to target

        # sets in target_mode that intersect with target_state
        sets_with_target = self.get_sets_in_mode_that_intersect_with_set(
            self.target_mode,
            self.name_to_vertex["target"].set(),
            just_one=self.opt.connect_source_target_to_single_set,
        )
        names_of_sets_with_target = []
        # at each horizon level, only sets that contain the target can transition into target
        # for layer in range(self.opt.horizon):
        for layer in (self.opt.horizon - 1,):
            # if that layer has a target mode
            if self.target_mode in self.modes_per_layer[layer]:
                # for each set that contains the target
                for set_id in sets_with_target:
                    names_of_sets_with_target += [self.get_vertex_name(layer, set_id)]
        # add the edges
        self.connect_to_vertex_on_the_left(
            names_of_sets_with_target, "target", EdgeOptions.within_mode_edge()
        )

    ###################################################################################
    # Vertex and edge naming

    def get_vertex_name(self, layer: int, set_id: int, t="M") -> str:
        """Naming convention is: M_<layer>_<set_id> for regular nodes"""
        return t + "_" + str(layer) + "_" + str(set_id)

    def get_edge_name(self, left_vertex_name: str, right_vertex_name: str) -> str:
        if right_vertex_name == "target":
            layer = int(left_vertex_name.split("_")[-2])
            return "Free move to target" # at " + str(layer)
        if left_vertex_name == "start":
            return "Equals start"

        layer = int(left_vertex_name.split("_")[-2])
        left_mode = self.get_mode_from_vertex_name(left_vertex_name)
        right_mode = self.get_mode_from_vertex_name(right_vertex_name)
        if left_mode in ("0", 0) and right_mode in ("0", 0):
            return "Free move"
        elif left_mode in ("0", 0):
            return "Grasp " + str(right_mode)
        else:
            return "Move, ungrasp " + str(left_mode) #+ " at " + str(layer)
        # return "E: " + left_vertex_name + " -> " + right_vertex_name
