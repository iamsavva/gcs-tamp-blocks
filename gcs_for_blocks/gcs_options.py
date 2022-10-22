class GCSforBlocksOptions:
    """
    Option class for GCSforBlocks.
    """

    block_dim: int  # number of dimensions that describe the block world
    num_blocks: int  # number of blocks
    horizon: int  # the GCS is ran in a receding horizon style (akin to a trellis diagram)

    block_width: float = 1.0  # block width

    # connectivity between modes: whether i -> i edges are added.
    allow_self_transitions_for_modes: bool = False
    # allow-self-transitions    -- allow transitioning into itself
    # no-self-transitions       -- don't allow transitioning into itself

    # add a time cost on each edge? this is done to "regularize" the trajectory
    # goal is to reduce possibility of pointlessly grasping and ungrasping in place
    add_time_cost: bool = True
    time_cost_weight: float = 1.0  # relative weight between

    # obstacle avoidance
    problem_complexity: str = "obstacles"
    # options:
    # transparent-no-obstacles  -- no collision avoidance, just block movement.
    # transparent-obstacles     -- collision avoidance with pre-defined obstacles but not other
    #                              blocks. not implemented yet
    # obstacles                 -- collision avoidance with both obstacles and other blocks.

    num_gcs_sets: int = -1

    # when solving, this is the max number of rounded paths
    max_rounded_paths: int = 50
    use_convex_relaxation: bool = True

    # for each mode, consider creating an unconstrained in/out node.
    # those this makes the problem significantly less tight, it reduces the number of edges.
    in_and_out_through_a_single_node = False

    # whether source and target ought to be connected to just one set in the mode
    # TODO: this should really be a default behavior always;
    # it reduces the number of edges and cycles
    connect_source_target_to_single_set: bool = True

    @property
    def num_modes(self) -> int:
        """
        Number of modes. For the case with no pushing, we have 1 mode for free motion and a mode
        per block for when grasping that block.
        The case with pushing will have many more modes; not implemented.
        """
        return self.num_blocks + 1

    @property
    def state_dim(self) -> int:
        """
        Dimension of the state x optimized at each vertex.
        (number of blocks + gripper) x (dimension of the world)
        """
        return (self.num_blocks + 1) * self.block_dim

    def __init__(
        self,
        block_dim: int,
        num_blocks: int,
        horizon: int,
        block_width: float = 1.0,
        allow_self_transitions_for_modes=False,
        add_time_cost: bool = True,
        time_cost_weight: float = 1.0,
        problem_complexity: str = "obstacles",
        max_rounded_paths: int = 40,
        use_convex_relaxation: bool = True,
        connect_source_target_to_single_set: bool = True,
        in_and_out_through_a_single_node: bool = False,
    ):
        assert problem_complexity in ("transparent-no-obstacles", "obstacles")
        self.block_dim = block_dim
        self.num_blocks = num_blocks
        self.horizon = horizon
        self.block_width = block_width
        self.allow_self_transitions_for_modes = allow_self_transitions_for_modes
        self.add_time_cost = add_time_cost
        self.time_cost_weight = time_cost_weight
        self.problem_complexity = problem_complexity
        self.max_rounded_paths = max_rounded_paths
        self.use_convex_relaxation = use_convex_relaxation
        self.connect_source_target_to_single_set = connect_source_target_to_single_set
        self.in_and_out_through_a_single_node = in_and_out_through_a_single_node

        self.num_gcs_sets = -1
