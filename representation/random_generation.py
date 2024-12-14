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
    edge_indices = torch.tensor([
        nodes_to_edge[n1, n2]
            for n1 in range(num_nodes)
                for n2 in range(n1+1, num_nodes)
    ])
    num_total_edges = edge_indices.shape[0]
    edge_perms = torch.stack([torch.randperm(num_total_edges) for _ in range(num_samples)])

    # Adds edges while there are not yet `num_edges` edges in each sample
    for i in range(num_total_edges):
        edge_adj[base_indices, edge_indices[edge_perms[:,i]]] += 1 * ((edge_adj > 0).sum(dim=1) < num_edges)
    
    return (edge_adj > 0).to(torch.float32)


def random_face_adjacencies(num_samples: int, num_nodes: int, num_faces: int) -> tuple[torch.Tensor]:
    """
    Generates random face adjacencies based on the given attributes.
    The face adjacencies are generated such that:
        1) There has no disconnected components.
        2) The material has the number of active nodes specified.

    num_samples: `int`
        The number of random truss samples to generate.

    num_nodes: `int`
        The exact number of nodes to use in every sample.
        Must be at least 3.

    num_faces: `torch.Tensor`
        The exact number of faces to use in each sample.
        Must be at least `num_nodes-2`.

    Returns: `torch.Tensor`
        A tuple of tensors containing the following tensors:
        
        A `(N,R)` tensor with the random face adjacencies.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.
        
        A `(N,R)` tensor with the corresponding edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
    """

    # Initializes the random face adjacencies
    face_adj = torch.zeros((num_samples,FACE_ADJ_SIZE))

    # Stores the face indices for each node
    nodes_to_face = torch.tensor([[[
        face_adj_index(n1, n2, n3)
            for n3 in range(num_nodes)]
                for n2 in range(num_nodes)]
                    for n1 in range(num_nodes)])

    # Prepares the values for creating one connected component
    node_perms = torch.stack([torch.randperm(num_nodes) for _ in range(num_samples)])
    base_indices = torch.arange(num_samples)

    # Stores the number of nodes that have been used for faces so far
    nodes_used = (torch.rand((num_samples,)) < 0.5) + 1 # Starts with 1,2

    # Stores the number of nodes to use at the next face addition
    new_nodes = 3 - nodes_used

    # Ensures each node is included in one connected component
    for n in range(2, num_nodes):
        
        ### FACE ON EXISTING VERTEX

        # Stores which samples will have a face at an existing face's vertex
        face_at_vertex = torch.nonzero(torch.logical_and((nodes_used + new_nodes) == (n+1), new_nodes == 2))[:,0]

        # Stores which old nodes will be chosen
        old_node_indices = torch.randint(0, n-1, (face_at_vertex.shape[0],))

        # Creates the face
        n1_index = node_perms[face_at_vertex, old_node_indices]
        n2_index = node_perms[face_at_vertex, n-1]
        n3_index = node_perms[face_at_vertex, n]
        face_index = nodes_to_face[n1_index, n2_index, n3_index]
        face_adj[face_at_vertex, face_index] = 1

        # Stores which kind of face will be the next face to be added
        nodes_used[face_at_vertex] += 2
        new_nodes[face_at_vertex] = (torch.rand((face_at_vertex.shape[0],)) < 0.5) + 1

        ### FACE ON EXISTING EDGE

        # Handles case when there is one node left but a face-on-vertex was chosen
        if n == num_nodes-1:
            new_nodes[:] = 1

        # Stores which samples will have a face at an existing face's edge
        face_at_edge = torch.nonzero(torch.logical_and((nodes_used + new_nodes) == (n+1), new_nodes == 1))[:,0]
        
        # Stores which old nodes will be chosen
        old_node_indices = torch.stack([torch.randperm(n) for _ in range(face_at_edge.shape[0])])[:,:2]

        # Creates the face
        n1_index = node_perms[face_at_edge, old_node_indices[:,0]]
        n2_index = node_perms[face_at_edge, old_node_indices[:,1]]
        n3_index = node_perms[face_at_edge, n]
        face_index = nodes_to_face[n1_index, n2_index, n3_index]
        face_adj[face_at_edge, face_index] = 1

        # Stores which kind of face will be the next face to be added
        nodes_used[face_at_edge] += 1
        new_nodes[face_at_edge] = (torch.rand((face_at_edge.shape[0],)) < 0.5) + 1

    # Computes useful values for adding the remaining faces
    face_indices = torch.tensor([
        nodes_to_face[n1, n2, n3]
            for n1 in range(num_nodes)
                for n2 in range(n1+1, num_nodes)
                    for n3 in range(n2+1, num_nodes)
    ])
    num_total_faces = face_indices.shape[0]
    face_perms = torch.stack([torch.randperm(num_total_faces) for _ in range(num_samples)])

    # Adds face while there are not yet `num_faces` faces in each sample
    for i in range(num_total_faces):
        face_adj[base_indices, face_indices[face_perms[:,i]]] += 1 * ((face_adj > 0).sum(dim=1) < num_faces)
    face_adj = (face_adj > 0).to(torch.float32)

    # Maps face adjacency indices to the corresponding nodes
    face_to_nodes = torch.tensor([
        [n1, n2, n3]
            for n1 in range(NUM_NODES)
                for n2 in range(n1+1, NUM_NODES)
                    for n3 in range(n2+1, NUM_NODES)
    ])

    # Computes the corresponding edge adjacency
    edge_adj = torch.zeros((num_samples, EDGE_ADJ_SIZE))
    for n1 in range(NUM_NODES):
        for n2 in range(NUM_NODES):
            if n1 == n2:
                continue
            for n3 in range(NUM_NODES):
                if n1 == n3 or n2 == n3:
                    continue
                has_face = face_adj[:, face_adj_index(n1,n2,n3)]
                edge_adj[:, edge_adj_index(n1,n2)] += has_face
                edge_adj[:, edge_adj_index(n1,n3)] += has_face
                edge_adj[:, edge_adj_index(n2,n3)] += has_face
    edge_adj = (edge_adj > 0).to(torch.float32)
    
    return face_adj, edge_adj


def straight_edge_parameters(node_coords: torch.Tensor, edge_adj: torch.Tensor) -> torch.Tensor:
    """
    Computes the edge parameters for straight edges for
    the given samples' node coordinates.

    node_coords: `torch.Tensor`
        A `(N,R//3,3)` tensor with the samples' node positions
        transformed into Euclidean coordinates.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.

    edge_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the straight edge parameters.
        Contains only edge parameters for the given active edges.
        All other edges parameters are 0.
        `N` is the number of samples.
        `R` is the size of the edge parameters array in the
        Metamaterial representation.
    """

    # Stores the node indices for each edge
    node_indices = torch.tensor([[n1, n2] for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES)])

    # Stores the vector describing each edge
    edge_vectors = node_coords[:,node_indices[:,1]] - node_coords[:,node_indices[:,0]]

    # Stores the edge parameters that make a straight edge
    edge_params = torch.cat([edge_vectors/3, 2*edge_vectors/3], dim=2) * edge_adj.unsqueeze(-1)
    edge_params = edge_params.reshape((node_coords.shape[0],-1))

    return edge_params


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

    # Stores the edge parameters
    edge_params = straight_edge_parameters(node_coords, edge_adj)

    # Stores the other parameters
    face_adj = torch.zeros((num_samples,FACE_ADJ_SIZE))
    face_params = torch.zeros((num_samples,FACE_PARAMS_SIZE))
    global_params = torch.ones((num_samples,1)) * 0.5 # Thickness

    return torch.cat([node_pos, edge_adj, edge_params, face_adj, face_params, global_params], dim=1)


def random_shells(num_samples: int, num_nodes: int, num_faces: int):
    """
    Generates random shell metamaterials based on the given attributes.
    The metamaterials are generated such that:
        1) There has no disconnected components.
        2) Every face in the unit cube has at least one node.
        3) The material has the number of active nodes specified.
        4) The material is a shell (i.e., has only flat faces).
        5) All nodes are on at least one face of the unit cube.
        6) All nodes are on at most three faces of the unit cube.

    num_samples: `int`
        The number of random shell samples to generate.

    num_nodes: `int`
        The exact number of nodes to use in every sample.
        Must be at least 3.

    num_faces: `torch.Tensor`
        The exact number of edges to use in each sample.
        Must be at least `num_nodes-2`.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the random samples of shell
        metamaterials.
        `N` is the number of samples.
        `R` is the representation size.
    """

    # Checks for valid parameters
    assert type(num_samples) == int
    assert type(num_nodes) == int
    assert type(num_faces) == int
    assert num_samples > 0, "Must generate at least one sample."
    assert num_nodes >= 3, f"No faces can be made with {num_nodes} nodes."
    assert num_nodes <= NUM_NODES, f"The representation cannot handle {num_nodes} nodes with its chosen hyperparameters."
    assert num_nodes * (num_nodes-1) * (num_nodes-2) // 6 >= num_faces >= num_nodes-2, f"No shell with {num_nodes} nodes can be made with {num_faces} faces."

    # Stores the node positions
    node_pos, node_coords = random_node_positions(num_samples, num_nodes)

    # Stores the edge/face adjacencies
    face_adj, edge_adj = random_face_adjacencies(num_samples, num_nodes, num_faces)

    # Stores the edge parameters
    edge_params = straight_edge_parameters(node_coords, edge_adj)


    ### FACE PARAMETERS

    # Stores the node indices for each edge
    node_indices = torch.tensor([
        [n1, n2, n3]
            for n1 in range(NUM_NODES)
                for n2 in range(n1+1, NUM_NODES)
                    for n3 in range(n2+1, NUM_NODES)
    ])

    # Stores the face parameters that make a flat face
    face_params = node_coords[:, node_indices].sum(dim=2) / 3 - node_coords[:, node_indices[:,0]]
    face_params = (face_params * face_adj.unsqueeze(-1)).reshape((num_samples,-1))


    # Stores the other parameters
    global_params = torch.ones((num_samples,1)) * 0.5 # Thickness

    return torch.cat([node_pos, edge_adj, edge_params, face_adj, face_params, global_params], dim=1)

