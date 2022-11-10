from gcs_for_blocks.set_tesselation import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions

import numpy as np

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    HPolyhedron,
    ConvexSet,
)

if __name__ == "__main__":
    dim = 2
    nb = 3
    h = 3
    options = GCSforAutonomousBlocksOptions(dim, nb, h)
    # st_red = SetTesselation(options, True)
    st = SetTesselation(options)

    print(len(st.dir2set))
    print(len(st.sets_in_dir_representation))
    print(len(st.dir2set) / len(st.sets_in_dir_representation))
