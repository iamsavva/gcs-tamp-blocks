from gcs_for_blocks.set_tesselation import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions

if __name__ == "__main__":
    dim = 2
    nb = 4
    h = 3
    options = GCSforAutonomousBlocksOptions(dim, nb, h)
    st = SetTesselation(options)
    print(len(st.dir2set))
    print(len(st.sets_in_dir_representation))
    print(len(st.dir2set)/len(st.sets_in_dir_representation))