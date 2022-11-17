from test import (
    make_simple_obstacle_swap_two,
    make_some_simple_transparent_tests,
    make_simple_obstacle_swap_two,
)

from gcs_for_blocks.set_tesselation_2d import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions

if __name__ == "__main__":
    make_simple_obstacle_swap_two()
    make_some_simple_transparent_tests()

    options = GCSforAutonomousBlocksOptions(2, ubf=4)
    st = SetTesselation(options)
    A = st.gen_set_from_dir_rep("A")
    B = st.gen_set_from_dir_rep("B")
    L = st.gen_set_from_dir_rep("L")
    R = st.gen_set_from_dir_rep("R")
    assert A.IntersectsWith(B) is False
    assert A.IntersectsWith(L) is True
    assert A.IntersectsWith(R) is True
    assert B.IntersectsWith(L) is True
    assert B.IntersectsWith(R) is True
    assert L.IntersectsWith(R) is False
