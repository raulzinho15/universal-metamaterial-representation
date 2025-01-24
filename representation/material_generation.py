import torch
from representation.rep_utils import *

# Stores the device on which operations will be done
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Computes the node IDs corresponding to each edge, in order of edge IDs
NODE_EDGE_INDICES = torch.tensor([
    [n1, n2]
        for n1 in range(NUM_NODES)
            for n2 in range(n1+1, NUM_NODES)
], device=DEVICE)

# Stores the face combinations that are compatible for a node to simultaneously be on
COMPATIBLE_FACES = torch.tensor([
    [0, 1, 2], [0, 1, 5], [0, 4, 2], [0, 4, 5], 
    [3, 1, 2], [3, 1, 5], [3, 4, 2], [3, 4, 5],
    [0, 2, 1], [0, 2, 4], [0, 5, 1], [0, 5, 4],
    [3, 2, 1], [3, 2, 4], [3, 5, 1], [3, 5, 4],
    [1, 0, 2], [1, 0, 5], [1, 3, 2], [1, 3, 5],
    [4, 0, 2], [4, 0, 5], [4, 3, 2], [4, 3, 5],
    [1, 2, 0], [1, 2, 3], [1, 5, 0], [1, 5, 3],
    [4, 2, 0], [4, 2, 3], [4, 5, 0], [4, 5, 3],
    [2, 0, 1], [2, 0, 4], [2, 3, 1], [2, 3, 4],
    [5, 0, 1], [5, 0, 4], [5, 3, 1], [5, 3, 4],
    [2, 1, 0], [2, 1, 3], [2, 4, 0], [2, 4, 3],
    [5, 1, 0], [5, 1, 3], [5, 4, 0], [5, 4, 3],
], device=DEVICE)
NUM_FACE_COMBINATIONS = COMPATIBLE_FACES.shape[0]


def generate_random_permutations(draws: int, max_values: torch.Tensor) -> torch.Tensor:
    """
    Generates `draws`-long random permutations of the numbers between 0 (inclusive)
    and `max_values` (exclusive). It must be that `draws <= max_values.max()`.
    """

    # Computes all the permutations
    flat_max_values = max_values.flatten()
    num_samples = flat_max_values.shape[0]
    all_perms = torch.rand((num_samples, max_values.max().item()), device=DEVICE).argsort(dim=-1)+1

    # Masks the values above the threshold
    mask = all_perms <= flat_max_values.unsqueeze(-1)

    # Computes the permutations
    rows = torch.arange(num_samples, device=DEVICE).reshape(-1,1).expand((num_samples, draws))
    flat_permutations: torch.Tensor = all_perms[rows, mask.argsort(dim=-1, descending=True)[:, :draws]]-1

    return flat_permutations.reshape((*max_values.shape, draws))


def generate_node_positions(num_nodes: torch.Tensor) -> torch.Tensor:
    """
    Generates random node position samples, where the
    number of samples is the same as the number of
    values in the `num_nodes` input.
    Excluded nodes' positions will be set to the cube center.

    The node positions are generated such that:
        1. Every face in the unit cube has at least one node.
        2. All nodes are on at most three faces of the unit cube.
        3. All nodes are at least about 0.14 units apart.
        4. All extra nodes are placed at the unit cube center.

    num_nodes: `torch.Tensor`
        A `(N,)` int tensor with the number of nodes for each node position
        sample. Each value must be at least 2, and at most `NUM_NODES`.
        `N` is the number of samples.

    Returns: `tuple[torch.Tensor, torch.Tensor]`
        A tuple of tensors containing the following tensors:
        
        A `(N,R)` tensor with the random pseudo-spherical
        node positions.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.
        
        A `(N,R//3,3)` tensor with the random node positions
        transformed into Euclidean coordinates.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.
    """

    # Performs safety checks
    assert num_nodes.numel() > 0, "Must have at least one node count."
    assert len(num_nodes.shape) == 1, "num_nodes must be a 1D tensor."
    assert num_nodes.dtype == torch.int32 or num_nodes.dtype == torch.int64, "num_nodes must be an int tensor."
    assert num_nodes.max().item() <= NUM_NODES, "num_nodes cannot contain values above NUM_NODES."
    assert num_nodes.min().item() >= 2, "num_nodes cannot contain values below 2."

    # Stores relevant & convenient values for the function
    num_samples = num_nodes.shape[0]
    samples_per_group = num_samples
    grid_size = 7
    assert NUM_NODES <= 10, "Current value of grid_size may not work"

    # Stores the grid offset indices
    grid_offsets = torch.stack(torch.meshgrid([torch.tensor([-1, 0, 1], device=DEVICE)]*3, indexing="ij")).reshape((3,-1)).unsqueeze(0)
    prob_indices = torch.tensor([[0] + [i for i in range(grid_size)] + [grid_size-1]], dtype=torch.int32, device=DEVICE)

    # Will store the random node Euclidean coordinates
    node_coords = torch.full((num_samples,NUM_NODES,3), 0.5, dtype=torch.float32, device=DEVICE)

    # Runs through each sample group
    for sample in range(0, num_samples, samples_per_group):
        end_sample = min(num_samples, sample+samples_per_group)
        group_size = end_sample-sample

        # Constructs the possible places to put points
        grid = torch.ones((group_size, *([grid_size]*3)), dtype=torch.bool, device=DEVICE)

        # Stores the number of nodes left
        nodes_left = num_nodes[sample:end_sample].clone()
        max_node_count = nodes_left.max().item()

        # Stores which faces are empty
        empty_faces = torch.ones((group_size,3*(grid_size+2)), dtype=torch.int32, device=DEVICE)

        # Runs through each node
        for node in range(max_node_count):

            # Updates the data structures according to which samples are still being generated
            sample_indices = torch.nonzero(nodes_left > 0).squeeze(1)
            samples_left = sample_indices.shape[0]


            ### X FACES

            # Stores which faces are empty
            x0_empty, x1_empty = empty_faces[sample_indices,0], empty_faces[sample_indices,-3]

            # Computes the x face probabilities
            x_probs = grid[sample_indices.unsqueeze(-1),prob_indices].any(dim=(-2,-1)).to(torch.float32) # Checks which faces are still open
            x_probs[:, 0] *= grid_size*torch.logical_not(torch.logical_and(x1_empty, nodes_left[sample_indices] == 1)) # Disables x=0 when x=1 must happen, and scales the prob
            x_probs[:,-1] *= grid_size*torch.logical_not(torch.logical_and(x0_empty, nodes_left[sample_indices] == 1)) # Disables x=1 when x=0 must happen, and scales the prob
            x_probs[:,[ 1, 2]] *= 1-x0_empty.unsqueeze(-1) # Disables the free choices affecting x=0 when x=0 is still empty
            x_probs[:,[-3,-2]] *= 1-x1_empty.unsqueeze(-1) # Disables the free choices affecting x=1 when x=1 is still empty
            x_probs[:,1:-1] *= (nodes_left[sample_indices] > (x0_empty+x1_empty)).unsqueeze(-1) # Disables free choices when a face must be chosen

            # Chooses the placements
            x_placements = torch.multinomial(x_probs, 1).squeeze(1)
            x0_placement = x_placements == 0
            x1_placement = x_placements == grid_size+1
            x_free_placement = torch.logical_not(torch.logical_or(x0_placement, x1_placement))

            # Computes the indices/coordinates due to the placements (x=0 is automatically 0)
            x_indices = (
                x_free_placement * (x_placements-1) +
                x1_placement * (grid_size-1)
            )
            x_coords = (
                x_free_placement * ((x_placements-1 + torch.rand((samples_left,), device=DEVICE)) / grid_size) +
                x1_placement.to(torch.float32)
            )


            ### Y FACES

            # Stores which faces are empty
            y0_empty, y1_empty = empty_faces[sample_indices,1], empty_faces[sample_indices,-2]

            # Computes the y face probabilities
            y_probs = torch.cat([grid[sample_indices,x_indices,:1], grid[sample_indices,x_indices], grid[sample_indices,x_indices,-1:]], dim=1).any(dim=-1).to(torch.float32) # Checks which faces are still open
            y_probs[:, 0] *= grid_size*torch.logical_not(torch.logical_and(y1_empty, nodes_left[sample_indices] == 1)) # Disables y=0 when y=1 must happen, and scales the prob
            y_probs[:,-1] *= grid_size*torch.logical_not(torch.logical_and(y0_empty, nodes_left[sample_indices] == 1)) # Disables y=1 when y=0 must happen, and scales the prob
            y_probs[:,[ 1, 2]] *= 1-y0_empty.unsqueeze(-1) # Disables the free choices affecting y=0 when y=0 is still empty
            y_probs[:,[-3,-2]] *= 1-y1_empty.unsqueeze(-1) # Disables the free choices affecting y=1 when y=1 is still empty
            y_probs[:,1:-1] *= (nodes_left[sample_indices] > (y0_empty+y1_empty)).unsqueeze(-1) # Disables free choices when a face must be chosen

            # Chooses the placements
            y_placements = torch.multinomial(y_probs, 1).squeeze(1)
            y0_placement = y_placements == 0
            y1_placement = y_placements == grid_size+1
            y_free_placement = torch.logical_not(torch.logical_or(y0_placement, y1_placement))

            # Computes the indices/coordinates due to the placements (y=0 is automatically 0)
            y_indices = (
                y_free_placement * (y_placements-1) +
                y1_placement * (grid_size-1)
            )
            y_coords = (
                y_free_placement * ((y_placements-1 + torch.rand((samples_left,), device=DEVICE)) / grid_size) +
                y1_placement.to(torch.float32)
            )


            ### Z FACES

            # Stores which faces are empty
            z0_empty, z1_empty = empty_faces[sample_indices,2], empty_faces[sample_indices,-1]

            # Computes the z face probabilities
            z_probs = torch.cat([grid[sample_indices,x_indices,y_indices,:1], grid[sample_indices,x_indices,y_indices], grid[sample_indices,x_indices,y_indices,-1:]], dim=1).to(torch.float32) # Checks which faces are still open
            z_probs[:, 0] *= grid_size*torch.logical_not(torch.logical_and(z1_empty, nodes_left[sample_indices] == 1)) # Disables z=0 when z=1 must happen, and scales the prob
            z_probs[:,-1] *= grid_size*torch.logical_not(torch.logical_and(z0_empty, nodes_left[sample_indices] == 1)) # Disables z=1 when z=0 must happen, and scales the prob
            z_probs[:,[ 1, 2]] *= 1-z0_empty.unsqueeze(-1) # Disables the free choices affecting z=0 when z=0 is still empty
            z_probs[:,[-3,-2]] *= 1-z1_empty.unsqueeze(-1) # Disables the free choices affecting z=1 when z=1 is still empty
            z_probs[:,1:-1] *= (nodes_left[sample_indices] > (z0_empty+z1_empty)).unsqueeze(-1) # Disables free choices when a face must be chosen
            
            # Chooses the placements
            z_placements = torch.multinomial(z_probs, 1).squeeze(1)
            z0_placement = z_placements == 0
            z1_placement = z_placements == grid_size+1
            z_free_placement = torch.logical_not(torch.logical_or(z0_placement, z1_placement))

            # Computes the indices/coordinates due to the placements (z=0 is automatically 0)
            z_indices = (
                z_free_placement * (z_placements-1) +
                z1_placement * (grid_size-1)
            )
            z_coords = (
                z_free_placement * ((z_placements-1 + torch.rand((samples_left,), device=DEVICE)) / grid_size) +
                z1_placement.to(torch.float32)
            )


            # Updates the empty faces
            empty_faces[sample_indices,   x_placements*3] = torch.clamp(empty_faces[sample_indices,   x_placements*3]-1, min=0)
            empty_faces[sample_indices, 1+y_placements*3] = torch.clamp(empty_faces[sample_indices, 1+y_placements*3]-1, min=0)
            empty_faces[sample_indices, 2+z_placements*3] = torch.clamp(empty_faces[sample_indices, 2+z_placements*3]-1, min=0)

            # Updates the grid
            base_indices = torch.stack([x_indices, y_indices, z_indices], dim=-1).unsqueeze(-1)
            grid_indices = torch.clamp(grid_offsets + base_indices, min=0, max=grid_size-1)
            grid[sample_indices.unsqueeze(-1), grid_indices[:,0], grid_indices[:,1], grid_indices[:,2]] = False

            # Updates the node coordinates
            node_coords[sample_indices,node] = torch.stack([x_coords, y_coords, z_coords], dim=-1)

            # Updates the number of nodes left
            nodes_left -= 1
        
    # Permutes the node coordinate dimensions to ensure uniformity in the random choices
    dim_permutation = torch.rand((num_samples,3), device=DEVICE).argsort(dim=-1)
    rows = torch.arange(num_samples, device=DEVICE).unsqueeze(-1).expand((num_samples, 3))
    node_coords[rows, :, dim_permutation] = node_coords.transpose(1,2).clone()

    # Computes the pseudo-spherical positions
    node_pos = euclidean_to_pseudo_spherical_torch(node_coords).reshape((num_samples,-1))

    # Returns the computed values
    return node_pos, node_coords
