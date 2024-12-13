import torch
import numpy as np
import random
from representation.rep_utils import *

# Stores the face combinations that are compatible
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
])
NUM_FACE_COMBINATIONS = COMPATIBLE_FACES.shape[0]


def random_node_positions(num_samples: int, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random node positions based on the given attributes.
    Excluded nodes' positions will be set to the cube center.
    The node positions are generated such that:
        1) Every face in the unit cube has at least one node.
        2) All nodes are on at least one face of the unit cube.
        3) All nodes are on at most three faces of the unit cube.

    num_samples: `int`
        The number of random node position samples to generate.

    num_nodes: `int`
        The exact number of nodes to use in every sample.
        Must be at least 2.

    Returns: `torch.Tensor`
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

    # Will store the random node positions
    node_pos = torch.zeros((0,NODE_POS_SIZE//3,3))
    node_coords = torch.zeros((0,NODE_POS_SIZE//3,3))

    # Stores values for indexing
    rows = torch.arange(num_samples).view(-1, 1).expand(num_samples, num_nodes)
    cols = torch.arange(num_nodes).view(1, -1).expand(num_samples, num_nodes)

    # Generates new node positions until enough are valid
    while node_pos.shape[0] != num_samples:
        new_node_coords = torch.rand((num_samples,NODE_POS_SIZE//3,3))

        # Sets unused nodes to (0,0,0) in pseudo-spherical coordinates
        new_node_coords[:,num_nodes:] = 0.5

        # Chooses the face combination for each node
        combination_indices = torch.randint(0, NUM_FACE_COMBINATIONS, (num_samples,num_nodes))
        face_combinations = COMPATIBLE_FACES[combination_indices]

        # Ensures that each node is on at least one face and at most 3 faces
        face_probs = [1.0, 0.25, 0.05]
        for face in range(3):

            # Stores the old node position value
            old_val = new_node_coords[rows, cols, face_combinations[:,:,face]%3]

            # Stochastically clamps the node to the face
            probs = torch.rand((num_samples, num_nodes)) < face_probs[face]
            new_node_coords[rows, cols, face_combinations[:,:,face]%3] = (
                old_val * torch.logical_not(probs) + (face_combinations[:,:,face] // 3) * probs
            )

        # Ensures that each face has at least one node
        valid_nodes = new_node_coords[:,:,[0,1,2,0,1,2]] == torch.tensor([[[0,0,0,1,1,1]]])
        valid_nodes = torch.nonzero(valid_nodes.any(dim=1).all(dim=1))[:,0]

        # Stores only the valid samples
        cutoff = num_samples-node_pos.shape[0]
        new_node_coords = new_node_coords[valid_nodes][:cutoff]

        # Stores all the valid samples together
        node_coords = torch.cat([node_coords, new_node_coords])
        node_pos = torch.cat([node_pos, euclidean_to_pseudo_spherical_torch(new_node_coords)])

    # Properly flattens the node position
    node_pos = node_pos.reshape((num_samples,-1))

    return node_pos, node_coords


def random_edge_adjacencies(num_samples: int, num_nodes: int, num_edges: int) -> torch.Tensor:
    """
    Generates random edge adjacencies based on the given attributes.
    The edge adjacencies are generated such that:
        1) There has no disconnected components.
        2) The material has the number of active nodes specified.

    num_samples: `int`
        The number of random truss samples to generate.

    num_nodes: `int`
        The exact number of nodes to use in every sample.
        Must be at least 2.

    num_edges: `torch.Tensor`
        The exact number of edges to use in each sample.
        Must be at least `num_nodes-1`.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the random edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
    """

    # Initializes the random edge adjacencies
    edge_adj = torch.zeros((num_samples,EDGE_ADJ_SIZE))

    # Stores the edge indices for each node
    nodes_to_edge = torch.tensor([[edge_adj_index(n1, n2) for n2 in range(num_nodes)] for n1 in range(num_nodes)])

    # Ensures each node is included in one connected component
    node_perms = torch.stack([torch.randperm(num_nodes) for _ in range(num_samples)])
    base_indices = torch.arange(num_samples)
    for n in range(1, num_nodes):

        # Stores the nodes that will have an edge between them
        this_node = node_perms[:, n]
        other_node = node_perms[base_indices, torch.randint(0, n, (num_samples,))]

        # Places an edge between the two nodes
        edge_index = nodes_to_edge[this_node, other_node]
        edge_adj[base_indices, edge_index] = 1

    # Computes useful values for adding the remaining edges
    num_total_edges = num_nodes * (num_nodes-1) // 2
    edge_perms = torch.stack([torch.randperm(num_total_edges) for _ in range(num_samples)])
    edge_indices = torch.tensor([nodes_to_edge[n1, n2] for n1 in range(num_nodes) for n2 in range(n1+1, num_nodes)])

    # Adds edges while there are not yet `num_edges` edges in each sample
    for i in range(num_total_edges):
        edge_adj[base_indices,edge_indices[edge_perms[:,i]]] += 1 * ((edge_adj > 0).sum(dim=1) < num_edges)
    
    return (edge_adj > 0).to(torch.float32)



def random_trusses(num_samples: int, num_nodes: int, num_edges: int):
    """
    Generates random truss metamaterials based on the given attributes.
    The metamaterials are generated such that:
        1) There has no disconnected components.
        2) Every face in the unit cube has at least one node.
        3) The material has the number of active nodes specified.
        4) The material is a truss (i.e., has only flat edges).
        5) All nodes are on at least one face of the unit cube.
        6) All nodes are on at most three faces of the unit cube.

    num_samples: `int`
        The number of random truss samples to generate.

    num_nodes: `int`
        The exact number of nodes to use in every sample.
        Must be at least 2.

    num_edges: `torch.Tensor`
        The exact number of edges to use in each sample.
        Must be at least `num_nodes-1`.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the random samples of truss
        metamaterials.
        `N` is the number of samples.
        `R` is the representation size.
    """

    # Checks for valid parameters
    assert type(num_samples) == int
    assert type(num_nodes) == int
    assert type(num_edges) == int
    assert num_samples > 0, "Must generate at least one sample."
    assert num_nodes >= 2, f"No edges can be made with {num_nodes} nodes."
    assert num_nodes <= NUM_NODES, f"The representation cannot handle {num_nodes} nodes with its chosen hyperparameters."
    assert num_nodes * (num_nodes - 1) // 2 >= num_edges >= num_nodes-1, f"No truss with {num_nodes} nodes can be made with {num_edges} edges."


    # Stores the node positions
    node_pos, node_coords = random_node_positions(num_samples, num_nodes)

    # Stores the edge adjacencies
    edge_adj = random_edge_adjacencies(num_samples, num_nodes, num_edges)


    ### EDGE PARAMETERS

    # Stores the node indices for each edge
    node_indices = torch.tensor([[n1, n2] for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES)])

    # Stores the vector describing each edge
    edge_vectors = node_coords[:,node_indices[:,1]] - node_coords[:,node_indices[:,0]]

    # Stores the edge parameters that make a straight edge
    edge_params = torch.cat([edge_vectors/3, 2*edge_vectors/3], dim=2) * edge_adj.unsqueeze(-1)
    edge_params = edge_params.reshape((num_samples,-1))


    # Stores the other parameters
    face_adj = torch.zeros((num_samples,FACE_ADJ_SIZE))
    face_params = torch.zeros((num_samples,FACE_PARAMS_SIZE))
    global_params = torch.ones((num_samples,1)) * 0.5 # Thickness

    return torch.cat([node_pos, edge_adj, edge_params, face_adj, face_params, global_params], dim=1)

