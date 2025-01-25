import torch
from representation.rep_utils import *

# Stores the device on which operations will be done
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Computes the node IDs corresponding to each edge, in order of edge IDs
EDGE_TO_NODES = torch.tensor([
    [n1, n2]
        for n1 in range(NUM_NODES)
            for n2 in range(n1+1, NUM_NODES)
], device=DEVICE)

# Computes the node IDs corresponding to each face, in order of face IDs
FACE_TO_NODES = torch.tensor([
    [n1, n2, n3]
        for n1 in range(NUM_NODES)
            for n2 in range(n1+1, NUM_NODES)
                for n3 in range(n2+1, NUM_NODES)
], device=DEVICE)

# Computes the edge IDs corresponding to each node pair
NODES_TO_EDGE = torch.tensor([[
    edge_adj_index(n1,n2)
        for n1 in range(NUM_NODES)]
            for n2 in range(NUM_NODES)
], device=DEVICE)

# Computes the face IDs corresponding to each node triplet
NODES_TO_FACE = torch.tensor([[[
    face_adj_index(n1,n2,n3)
        for n1 in range(NUM_NODES)]
            for n2 in range(NUM_NODES)]
                for n3 in range(NUM_NODES)
], device=DEVICE)


def generate_random_permutations(max_values: torch.Tensor) -> torch.Tensor:
    """
    Generates random permutations of the numbers between 0 (inclusive)
    and `max_values` (exclusive). Any remaining numbers are a permutation
    of the remaining numbers.
    """

    # Computes all the permutations
    flat_max_values = max_values.flatten()
    num_samples = flat_max_values.shape[0]
    all_perms = torch.rand((num_samples, max_values.max().item()), device=DEVICE).argsort(dim=-1)+1

    # Masks the values above the threshold
    mask = all_perms <= flat_max_values.unsqueeze(-1)

    # Computes the permutations
    draws = flat_max_values.max().item()
    rows = torch.arange(num_samples, device=DEVICE).reshape(-1,1).expand((num_samples, draws))
    flat_permutations: torch.Tensor = all_perms[rows, mask.argsort(dim=-1, descending=True)]-1

    return flat_permutations.reshape((*max_values.shape, draws))


def generate_random_ints(max_values: torch.Tensor, min_values: torch.Tensor=0) -> torch.Tensor:
    """
    Generates random integers between `min_values` (inclusive) and `max_values` (exclusive).
    """
    return torch.floor(torch.rand(max_values.shape[0], device=DEVICE) * (max_values - min_values) + min_values).to(torch.int32)


def choose_node_x_placements(nodes_left: torch.Tensor, empty_slots: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Chooses the placement of the next node for each sample along the x dimension.

    nodes_left: `torch.Tensor`
        A `(N,)` tensor with the number of nodes left for each sample.
        `N` is the number of samples.

    empty_slots: `torch.Tensor`
        A `(N,3G+6)` tensor with a 1 when a particular node placement has not
        been made before, and 0 otherwise.
        `N` is the number of samples.
        `G` is the grid size.

    grid: `torch.Tensor`
        A `(N,G,G,G)` tensor with which grid spaces are not taken by a
        node or are too close to a node.
        `N` is the number of samples.
        `G` is the grid size.

    Returns: `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
        A `(N,)` tensor with the x slot placements for each node.
        `N` is the number of samples.
        
        A `(N,)` tensor with the x index placements for each node.
        `N` is the number of samples.
        
        A `(N,)` tensor with the x coordinate placements for each node.
        `N` is the number of samples.
    """

    # Stores convenient values for the function
    samples_left = grid.shape[0]
    grid_size = grid.shape[1]
    prob_indices = torch.tensor([0] + [i for i in range(grid_size)] + [grid_size-1], dtype=torch.int32, device=DEVICE)

    # Stores which faces are empty
    x0_empty, x1_empty = empty_slots[:,0], empty_slots[:,-3]

    # Checks which faces are still open
    x_probs = grid[:,prob_indices].any(dim=(-2,-1)).to(torch.float32)

    # Disables x=(0,1) when x=(1,0) must happen, and scales the probability
    x_probs[:, 0] *= grid_size*torch.logical_not(torch.logical_and(x1_empty, nodes_left == 1))
    x_probs[:,-1] *= grid_size*torch.logical_not(torch.logical_and(x0_empty, nodes_left == 1))

    # Disables the free choices affecting x=(0,1) when x=(0,1) is still empty
    x_probs[:,[ 1, 2]] *= torch.logical_not(x0_empty).unsqueeze(-1)
    x_probs[:,[-3,-2]] *= torch.logical_not(x1_empty).unsqueeze(-1)

    # Disables free choices when a face must be chosen
    x_probs[:,1:-1] *= (nodes_left > (x0_empty+x1_empty)).unsqueeze(-1)

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

    return x_placements, x_indices, x_coords


def choose_node_y_placements(nodes_left: torch.Tensor, empty_slots: torch.Tensor, grid: torch.Tensor, x_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Chooses the placement of the next node for each sample along the y dimension.

    nodes_left: `torch.Tensor`
        A `(N,)` tensor with the number of nodes left for each sample.
        `N` is the number of samples.

    empty_slots: `torch.Tensor`
        A `(N,3G+6)` tensor with a 1 when a particular node placement has not
        been made before, and 0 otherwise.
        `N` is the number of samples.
        `G` is the grid size.

    grid: `torch.Tensor`
        A `(N,G,G,G)` tensor with which grid spaces are not taken by a
        node or are too close to a node.
        `N` is the number of samples.
        `G` is the grid size.
        
    x_indices: `torch.Tensor`
        A `(N,)` tensor with the x index placements for each node.
        `N` is the number of samples.

    Returns: `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
        A `(N,)` tensor with the y slot placements for each node.
        `N` is the number of samples.
        
        A `(N,)` tensor with the y index placements for each node.
        `N` is the number of samples.
        
        A `(N,)` tensor with the y coordinate placements for each node.
        `N` is the number of samples.
    """

    # Stores convenient values for the function
    samples_left = grid.shape[0]
    sample_indices = torch.arange(samples_left, device=DEVICE)
    grid_size = grid.shape[1]

    # Stores which faces are empty
    y0_empty, y1_empty = empty_slots[:,1], empty_slots[:,-2]

    # Checks which faces are still open
    y_probs = torch.cat([grid[sample_indices,x_indices,:1], grid[sample_indices,x_indices], grid[sample_indices,x_indices,-1:]], dim=1).any(dim=-1).to(torch.float32)

    # Disables y=(0,1) when y=(1,0) must happen, and scales the probability
    y_probs[:, 0] *= grid_size*torch.logical_not(torch.logical_and(y1_empty, nodes_left == 1))
    y_probs[:,-1] *= grid_size*torch.logical_not(torch.logical_and(y0_empty, nodes_left == 1))

    # Disables the free choices affecting y=(0,1) when y=(0,1) is still empty
    y_probs[:,[ 1, 2]] *= 1-y0_empty.unsqueeze(-1)
    y_probs[:,[-3,-2]] *= 1-y1_empty.unsqueeze(-1)

    # Disables free choices when a face must be chosen
    y_probs[:,1:-1] *= (nodes_left > (y0_empty+y1_empty)).unsqueeze(-1)

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

    return y_placements, y_indices, y_coords


def choose_node_z_placements(nodes_left: torch.Tensor, empty_slots: torch.Tensor, grid: torch.Tensor, x_indices: torch.Tensor, y_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Chooses the placement of the next node for each sample along the z dimension.

    nodes_left: `torch.Tensor`
        A `(N,)` tensor with the number of nodes left for each sample.
        `N` is the number of samples.

    empty_slots: `torch.Tensor`
        A `(N,3G+6)` tensor with a 1 when a particular node placement has not
        been made before, and 0 otherwise.
        `N` is the number of samples.
        `G` is the grid size.

    grid: `torch.Tensor`
        A `(N,G,G,G)` tensor with which grid spaces are not taken by a
        node or are too close to a node.
        `N` is the number of samples.
        `G` is the grid size.
        
    x_indices: `torch.Tensor`
        A `(N,)` tensor with the x index placements for each node.
        `N` is the number of samples.
        
    y_indices: `torch.Tensor`
        A `(N,)` tensor with the y index placements for each node.
        `N` is the number of samples.

    Returns: `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
        A `(N,)` tensor with the z slot placements for each node.
        `N` is the number of samples.
        
        A `(N,)` tensor with the z index placements for each node.
        `N` is the number of samples.
        
        A `(N,)` tensor with the z coordinate placements for each node.
        `N` is the number of samples.
    """

    # Stores convenient values for the function
    samples_left = grid.shape[0]
    sample_indices = torch.arange(samples_left, device=DEVICE)
    grid_size = grid.shape[1]

    # Stores which faces are empty
    z0_empty, z1_empty = empty_slots[:,2], empty_slots[:,-1]

    # Checks which faces are still open
    z_probs = torch.cat([grid[sample_indices,x_indices,y_indices,:1], grid[sample_indices,x_indices,y_indices], grid[sample_indices,x_indices,y_indices,-1:]], dim=1).to(torch.float32)

    # Disables z=(0,1) when z=(1,0) must happen, and scales the probability
    z_probs[:, 0] *= grid_size*torch.logical_not(torch.logical_and(z1_empty, nodes_left == 1))
    z_probs[:,-1] *= grid_size*torch.logical_not(torch.logical_and(z0_empty, nodes_left == 1))

    # Disables the free choices affecting z=(0,1) when z=(0,1) is still empty
    z_probs[:,[ 1, 2]] *= 1-z0_empty.unsqueeze(-1)
    z_probs[:,[-3,-2]] *= 1-z1_empty.unsqueeze(-1)

    # Disables free choices when a face must be chosen
    z_probs[:,1:-1] *= (nodes_left > (z0_empty+z1_empty)).unsqueeze(-1)
    
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

    return z_placements, z_indices, z_coords


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
        
        A `(N,R)` float tensor with the random pseudo-spherical
        node positions.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.
        
        A `(N,R//3,3)` float tensor with the random node positions
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

    # Will store the random node Euclidean coordinates
    node_coords = torch.full((num_samples,NUM_NODES,3), 0.5, device=DEVICE)

    # Runs through each sample group
    for sample in range(0, num_samples, samples_per_group):

        # Stores values for the sample group
        end_sample = min(num_samples, sample+samples_per_group)
        group_size = end_sample-sample
        old_indices = torch.arange(group_size, device=DEVICE)

        # Constructs the possible places to put points
        grid = torch.ones((group_size, *([grid_size]*3)), dtype=torch.bool, device=DEVICE)

        # Stores the number of nodes left
        nodes_left = num_nodes[sample:end_sample].clone()
        max_node_count = nodes_left.max().item()

        # Stores which faces are empty
        empty_faces = torch.ones((group_size,3*(grid_size+2)), dtype=torch.int32, device=DEVICE)

        # Runs through each node
        for node in range(max_node_count):

            # Stores the samples still being generated
            sample_indices = torch.nonzero(nodes_left > 0).squeeze(1)
            samples_left = sample_indices.shape[0]

            # Updates the data structures according to which samples are left
            nodes_left = nodes_left[sample_indices]
            empty_faces = empty_faces[sample_indices]
            grid = grid[sample_indices]
            old_indices = old_indices[sample_indices]
            sample_indices = torch.arange(samples_left, device=DEVICE)

            # Chooses the node placements
            x_placements, x_indices, x_coords = choose_node_x_placements(nodes_left, empty_faces, grid)
            y_placements, y_indices, y_coords = choose_node_y_placements(nodes_left, empty_faces, grid, x_indices)
            z_placements, z_indices, z_coords = choose_node_z_placements(nodes_left, empty_faces, grid, x_indices, y_indices)

            # Updates the empty faces
            empty_faces[sample_indices,   x_placements*3] = empty_faces[sample_indices,   x_placements*3]-1
            empty_faces[sample_indices, 1+y_placements*3] = empty_faces[sample_indices, 1+y_placements*3]-1
            empty_faces[sample_indices, 2+z_placements*3] = empty_faces[sample_indices, 2+z_placements*3]-1
            empty_faces.clamp_(min=0)

            # Updates the grid
            grid_indices = torch.stack([x_indices, y_indices, z_indices], dim=-1).unsqueeze(-1) + grid_offsets
            grid_indices.clamp_(min=0, max=grid_size-1)
            grid[sample_indices.unsqueeze(-1), grid_indices[:,0], grid_indices[:,1], grid_indices[:,2]] = False

            # Updates the node coordinates
            node_coords[old_indices+sample,node] = torch.stack([x_coords, y_coords, z_coords], dim=-1)

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


def generate_adjacencies(num_nodes: torch.Tensor, num_edges: torch.Tensor, num_faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random edge and face adjacencies such that the resulting topology
    of the metamaterial is connected.

    num_nodes: `torch.Tensor`
        A `(N,)` int tensor with the number of active nodes to use in each sample.
        `N` is the number of samples.

    num_edges: `torch.Tensor`
        A `(N,)` int tensor with the maximum number of non-face edges to use in
        each sample. The number is the maximum as the choice of faces is prioritized.
        If the choice of faces allows for this many non-face edges, then there will
        be this many edges. Otherwise, the maximum number of edges with the generated
        face arrangement will be chosen. There will always be at least this many edges.
        `N` is the number of samples.

    num_faces: `torch.Tensor`
        A `(N,)` int tensor with the total number of faces to use in each sample.
        `N` is the number of samples.

    Returns: `tuple[torch.Tensor, torch.Tensor]`
        A `(N,R)` tensor with the samples' edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
        
        A `(N,R)` tensor with the samples' face adjacencies.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.
    """

    # Performs safety checks
    assert num_nodes.numel() > 0, "Must have at least one node count."
    assert num_nodes.numel() == num_edges.numel(), "Must have the same number of node counts as edge counts."
    assert num_nodes.numel() == num_faces.numel(), "Must have the same number of node counts as face counts."
    assert len(num_nodes.shape) == 1, "num_nodes must be a 1D tensor."
    assert len(num_edges.shape) == 1, "num_edges must be a 1D tensor."
    assert len(num_faces.shape) == 1, "num_faces must be a 1D tensor."
    assert num_nodes.dtype == torch.int32 or num_nodes.dtype == torch.int64, "num_nodes must be an int tensor."
    assert num_edges.dtype == torch.int32 or num_edges.dtype == torch.int64, "num_edges must be an int tensor."
    assert num_faces.dtype == torch.int32 or num_faces.dtype == torch.int64, "num_faces must be an int tensor."
    assert num_nodes.max().item() <= NUM_NODES, "num_nodes cannot contain values above NUM_NODES."
    assert num_nodes.min().item() >= 2, "num_nodes cannot contain values below 2."
    assert torch.all(num_nodes <= 1 + num_edges + 2*num_faces).item(), "There are not enough edges and faces to make all nodes connected."
    max_edges = num_nodes * (num_nodes-1) // 2
    assert torch.all(num_edges <= max_edges).item(), "There are not enough nodes to make all the edges."
    max_faces = num_nodes * (num_nodes-1) * (num_nodes-2) // 6
    assert torch.all(num_faces <= max_faces).item(), "There are not enough nodes to make all the faces."

    # Stores relevant & convenient values for the function
    num_samples = num_nodes.shape[0]
    base_indices = torch.arange(num_samples, dtype=torch.int32, device=DEVICE)
    max_nodes = num_nodes.max().item()

    # Will store the edge and face adjancencies
    edge_adj = torch.zeros((num_samples, EDGE_ADJ_SIZE), device=DEVICE)
    face_adj = torch.zeros((num_samples, FACE_ADJ_SIZE), device=DEVICE)

    # Stores the random node permutation for edge/face connection
    node_orderings = generate_random_permutations(num_nodes)

    # Stores values for making connections
    old_indices = base_indices.clone()
    nodes_left = num_nodes.clone()-1
    edges_left = num_edges.clone()
    faces_left = num_faces.clone()

    # Runs through each node to make the base topology connection
    while True:

        # Stores the samples still being generated
        sample_indices = torch.nonzero(nodes_left > 0).squeeze(1)
        samples_left = sample_indices.shape[0]

        # Breaks out if no samples left
        if samples_left == 0:
            break

        # Updates the data structures storing info about the samples
        old_indices = old_indices[sample_indices]
        node_orderings = node_orderings[sample_indices]
        nodes_left = nodes_left[sample_indices]
        edges_left = edges_left[sample_indices]
        faces_left = faces_left[sample_indices]

        # Stores the current node to which a topology connection is being added
        current_node = num_nodes[old_indices]-nodes_left

        # Stores the probability of each action
        # - 0: new edge at an existing node
        # - 1: new face with one new node
        # - 2: new face with two new node
        actions_pdf = torch.stack([
            edges_left,
            faces_left * (nodes_left < faces_left*2 + edges_left) * (current_node > 1),
            faces_left * (nodes_left > 1),
        ], dim=-1).to(torch.float32)

        # Chooses the actions
        actions = torch.multinomial(actions_pdf, 1).squeeze(1)

        # Handles adding a new edge at an existing node
        material_indices = torch.nonzero(actions == 0).squeeze(1)
        if material_indices.numel() > 0:

            # Adjusts the number of nodes/edges left
            nodes_left[material_indices] -= 1
            edges_left[material_indices] -= 1

            # Chooses the nodes on which the edge addition will take place
            node1_index = generate_random_ints(current_node[material_indices])
            node1 = node_orderings[material_indices, node1_index]
            node2 = node_orderings[material_indices, current_node[material_indices]]

            # Adds the edge
            edge_adj[old_indices[material_indices], NODES_TO_EDGE[node1, node2]] = 1

        # Handles adding a new face with two new nodes
        material_indices = torch.nonzero(actions == 2).squeeze(1)
        if material_indices.numel() > 0:

            # Adjusts the number of nodes/faces left
            nodes_left[material_indices] -= 2
            faces_left[material_indices] -= 1

            # Chooses the nodes on which the face addition will take place
            node1_index = generate_random_ints(current_node[material_indices])
            node1 = node_orderings[material_indices, node1_index]
            node2 = node_orderings[material_indices, current_node[material_indices]]
            node3 = node_orderings[material_indices, current_node[material_indices]+1]

            # Adds the face
            face_adj[old_indices[material_indices], NODES_TO_FACE[node1, node2, node3]] = 1

        # Handles adding a new face with one new node
        material_indices = torch.nonzero(actions == 1).squeeze(1)
        if material_indices.numel() > 0:

            # Adjusts the number of nodes/faces left
            nodes_left[material_indices] -= 1
            faces_left[material_indices] -= 1

            # Chooses the nodes on which the face addition will take place
            node_choices = generate_random_permutations(current_node[material_indices])[:,:2]
            node1 = node_orderings[material_indices, node_choices[:,0]]
            node2 = node_orderings[material_indices, node_choices[:,1]]
            node3 = node_orderings[material_indices, current_node[material_indices]]

            # Adds the face
            face_adj[old_indices[material_indices], NODES_TO_FACE[node1, node2, node3]] = 1

    # Stores the face adjacency indices in order increasing largest node
    face_adj_indices = torch.tensor([
        face_adj_index(n1,n2,n3)
            for n3 in range(max_nodes)
                for n2 in range(n3-1, -1, -1)
                    for n1 in range(n2-1, -1, -1)
    ], device=DEVICE)

    # Stores a shuffling of the faces
    face_shuffle = generate_random_permutations(max_faces)

    # Fills in the rest of the faces
    for face in range(num_faces.max().item()):
        face_adj[base_indices, face_adj_indices[face_shuffle[:,face]]] += (
            (face_adj > 0).sum(dim=1) < num_faces
        )
    face_adj = (face_adj > 0).to(torch.float32)

    # Computes all the edges tied to an existing face
    face_edges = torch.zeros(edge_adj.shape, device=DEVICE)
    for n1 in range(max_nodes):
        for n2 in range(n1+1, max_nodes):
            for n3 in range(n2+1, max_nodes):

                # Stores the face index
                face_index = face_adj_index(n1,n2,n3)

                # Stores the edge_indices
                edge1_index = edge_adj_index(n1,n2)
                edge2_index = edge_adj_index(n1,n3)
                edge3_index = edge_adj_index(n2,n3)
                
                # Labels the edge as a face-edge
                face_val = face_adj[:,face_index]
                face_edges[:,edge1_index] += face_val
                face_edges[:,edge2_index] += face_val
                face_edges[:,edge3_index] += face_val
    
    # Normalizes the face-edge adjacency
    face_edges = (face_edges > 0).to(torch.float32)
    edge_offset = face_edges.sum(dim=1).to(torch.int32)
    edge_adj = ((edge_adj + face_edges) > 0).to(torch.int32)

    # Stores the edge adjacency indices in order increasing largest node
    edge_adj_indices = torch.tensor([
        edge_adj_index(n1,n2)
            for n2 in range(max_nodes)
                for n1 in range(n2-1, -1, -1)
    ], device=DEVICE)

    # Stores a shuffling of the edge
    edge_shuffle = generate_random_permutations(max_edges)

    # Fills in the rest of the edges
    edge_cutoff = min(max_edges.max().item(), (num_edges+edge_offset).max().item())
    for edge in range(edge_cutoff):
        edge_adj[base_indices, edge_adj_indices[edge_shuffle[:,edge]]] += (
            (edge_adj > 0).sum(dim=1) < (num_edges + edge_offset)
        )
    edge_adj = (edge_adj > 0).to(torch.float32)

    return edge_adj, face_adj




