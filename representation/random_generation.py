import torch
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
    edge_adj = (edge_adj > 0).to(torch.float32)

    # Shuffles the edge adjacencies
    node_perms = torch.stack([torch.randperm(num_nodes) for _ in range(num_samples)])
    new_edge_adj = torch.zeros(edge_adj.shape)
    for n1 in range(num_nodes):
        for n2 in range(n1+1, num_nodes):
            old_val = edge_adj[:, nodes_to_edge[n1,n2]]
            new_edge_adj[base_indices, nodes_to_edge[node_perms[:,n1], node_perms[:,n2]]] = old_val
    edge_adj = new_edge_adj

    return edge_adj


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
        Must be at least `(num_nodes-1)/2`.

    Returns: `tuple[torch.Tensor]`
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

    # Will store the valid random face adjacencies
    final_face_adj = torch.zeros((0,FACE_ADJ_SIZE))

    while final_face_adj.shape[0] != num_samples:

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

        # Stores the probability of a face at an existing face's vertex
        face_at_vertex_prob = 0.5 + 0.5 / (1 + num_faces - (num_nodes-1)//2)

        # Stores the number of nodes that have been used for faces so far
        nodes_used = (torch.rand((num_samples,)) > face_at_vertex_prob) + 1 # Starts with 1,2

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
            new_nodes[face_at_vertex] = (torch.rand((face_at_vertex.shape[0],)) < face_at_vertex_prob) + 1

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
            new_nodes[face_at_edge] = (torch.rand((face_at_edge.shape[0],)) < face_at_vertex_prob) + 1

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

        # Stores the valid face adjacencies
        valid_face_adj = face_adj[torch.nonzero(face_adj.sum(dim=1) == num_faces)[:,0]][:num_samples-final_face_adj.shape[0]]
        final_face_adj = torch.cat([final_face_adj, valid_face_adj])

    # Stores the finalized face adjacencies
    face_adj = final_face_adj

    # Shuffles the face adjacencies
    node_perms = torch.stack([torch.randperm(num_nodes) for _ in range(num_samples)])
    new_face_adj = torch.zeros(face_adj.shape)
    for n1 in range(num_nodes):
        for n2 in range(n1+1, num_nodes):
            for n3 in range(n2+1, num_nodes):
                old_val = face_adj[:, nodes_to_face[n1,n2,n3]]
                new_face_adj[base_indices, nodes_to_face[node_perms[:,n1], node_perms[:,n2], node_perms[:,n3]]] = old_val
    face_adj = new_face_adj

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


def generate_edge_and_face_adjacencies(num_samples: int, num_nodes: int, num_edges: int, num_faces: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random edge and face adjacencies such that the
    resulting topology of the metamaterial is connected.

    num_samples: `int`
        The number of random metamaterial samples to generate.

    num_nodes: `int`
        The number of active nodes to use in every sample.

    num_edges: `torch.Tensor`
        The total number of non-face edges to use in each sample.

    num_faces: `torch.Tensor`
        The total number of faces to use in each sample.

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

    # Will store the edge and face adjancencies
    edge_adj = torch.zeros((num_samples, EDGE_ADJ_SIZE))
    face_adj = torch.zeros((num_samples, FACE_ADJ_SIZE))

    # Stores the node index chain for the base edges
    node_orderings = torch.stack([torch.randperm(num_nodes) for _ in range(num_samples)])

    # Runs through each sample, creating the base connection between all the active nodes
    for sample in range(num_samples):

        # Stores values for making node connections
        current_node = 1
        nodes_left = num_nodes-1
        edges_left = num_edges
        faces_left = num_faces

        # Determines the sequence of connections to make
        while True:

            # Stores the next action to take
            # - 0: new edge at an existing vertex
            # - 1: new face at two existing vertices
            # - 2: new face at an existing vertex
            action_pdf = torch.tensor([
                edges_left,
                faces_left * (nodes_left < faces_left*2 + edges_left) * (current_node > 1),
                faces_left * (nodes_left > 1),
            ], dtype=torch.float32)
            assert action_pdf.sum().item() > 0, "The choice of base edge/face adjacencies is invalid"
            action_pdf /= action_pdf.sum().item()
            prob = random.random()

            # Adds a new edge at an existing vertex
            if prob < action_pdf[0]:
                edges_left -= 1
                nodes_left -= 1
                node1 = node_orderings[sample, random.randint(0,current_node-1)]
                node2 = node_orderings[sample, current_node]
                edge_adj[sample, edge_adj_index(node1, node2)] = 1
                current_node += 1

            # Adds a new face at an existing vertex
            elif 1-prob < action_pdf[2]:
                faces_left -= 1
                nodes_left -= 2
                node1 = node_orderings[sample, random.randint(0,current_node-1)]
                node2 = node_orderings[sample, current_node]
                node3 = node_orderings[sample, current_node+1]
                face_adj[sample, face_adj_index(node1, node2, node3)] = 1
                current_node += 2

            # Adds a new face at two existing vertices
            else:
                faces_left -= 1
                nodes_left -= 1
                node_choices = torch.randperm(current_node)[:2]
                node1 = node_orderings[sample, node_choices[0]]
                node2 = node_orderings[sample, node_choices[1]]
                node3 = node_orderings[sample, current_node]
                face_adj[sample, face_adj_index(node1, node2, node3)] = 1
                current_node += 1
            
            # Exits when no nodes are left
            if nodes_left == 0:
                break

    # Fills in the rest of the faces
    base_indices = torch.arange(num_samples)
    face_orderings = torch.stack([torch.randperm(FACE_ADJ_SIZE)[:num_faces] for _ in range(num_samples)])
    for face in range(num_faces):
        face_adj[base_indices, face_orderings[:,face]] += 1 * ((face_adj > 0).sum(dim=1) < num_faces)
    face_adj = (face_adj > 0).to(torch.float32)

    # Computes all the edges tied to an existing face
    face_edges = torch.zeros(edge_adj.shape)
    for n1 in range(num_nodes):
        for n2 in range(n1+1, num_nodes):
            for n3 in range(n2+1, num_nodes):

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

    # Fills in the rest of the edges
    edge_offset = face_edges.sum(dim=1)
    edge_orderings = torch.stack([torch.randperm(EDGE_ADJ_SIZE) for _ in range(num_samples)])
    for edge in range(EDGE_ADJ_SIZE):
        edge_adj[base_indices, edge_orderings[:,edge]] += 1 * ((edge_adj > 0).sum(dim=1) < num_edges + edge_offset)
    edge_adj = (edge_adj > 0).to(torch.float32)

    return edge_adj, face_adj


def flat_edge_parameters(node_coords: torch.Tensor, edge_adj: torch.Tensor) -> torch.Tensor:
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
        All other edge parameters are 0.
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


def flat_face_parameters(node_coords: torch.Tensor, face_adj: torch.Tensor) -> torch.Tensor:
    """
    Computes the face parameters for flat faces for
    the given samples' node coordinates.

    node_coords: `torch.Tensor`
        A `(N,R//3,3)` tensor with the samples' node positions
        transformed into Euclidean coordinates.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.

    face_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' face adjacencies.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the flat face parameters.
        Contains only face parameters for the given active faces.
        All other face parameters are 0.
        `N` is the number of samples.
        `R` is the size of the face parameters array in the
        Metamaterial representation.
    """

    # Stores the node indices for each edge
    node_indices = torch.tensor([
        [n1, n2, n3]
            for n1 in range(NUM_NODES)
                for n2 in range(n1+1, NUM_NODES)
                    for n3 in range(n2+1, NUM_NODES)
    ])

    # Stores the face parameters that make a flat face
    face_params = node_coords[:, node_indices].sum(dim=2) / 3 - node_coords[:, node_indices[:,0]]
    face_params = (face_params * face_adj.unsqueeze(-1)).reshape((node_coords.shape[0],-1))

    return face_params


def base_face_parameters(node_euclidean_coords: torch.Tensor, face_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the base face parameters for the given samples.

    node_euclidean_coords: `torch.Tensor`
        A `(N,R//3,3)` tensor with the Euclidean coordinates
        of each node, ordered in the second dimension by node ID.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.

    face_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' face adjacencies.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.

    Returns: `tuple[torch.Tensor]`
        A `(N,R,-1)` tensor with the samples' corresponding
        flat face parameters.

        A `(N,R,-1)` tensor with the samples' corresponding
        curved face parameters.

        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.
        The face parameters of non-existent faces are 0.
    """

    # Stores the number of metamaterial samples
    num_samples = node_euclidean_coords.shape[0]

    # Stores the general shape of the face parameter arrays
    face_params_shape = (num_samples, FACE_ADJ_SIZE, FACE_BEZIER_COORDS)

    # Computes the flat/curved face parameters
    flat_face_params = flat_face_parameters(node_euclidean_coords, face_adj).reshape(face_params_shape)
    curved_face_params = (torch.rand(face_params_shape)*2-1) * face_adj.unsqueeze(-1)

    return flat_face_params, curved_face_params


def choose_flat_and_curved_faces(num_curved_faces: int, face_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chooses which face will correspond to flat/curved faces
    in the given samples.

    num_curved_faces: `int`
        The number of curved faces to use in each sample.

    face_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' face adjacencies.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.

    Returns: `tuple[torch.Tensor]`
        A `(N,R)` tensor with the samples' face adjacencies
        just for the chosen flat faces.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.
        
        A `(N,R)` tensor with the samples' face adjacencies
        just for the chosen curved faces.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.
    """

    # Stores relevant values for the function
    num_samples = face_adj.shape[0]
    num_faces = int(face_adj[0].sum().item())

    # Stores the indices of active faces
    active_faces = torch.nonzero(face_adj)[:,1].reshape((num_samples, num_faces))

    # Generates shuffled indices
    shuffled_indices = torch.stack([torch.randperm(num_faces)[:num_curved_faces] for _ in range(num_samples)])
    rows = torch.arange(num_samples).view(-1, 1).expand(num_samples, num_curved_faces)

    # Chooses the flat/curved faces
    curved_faces = torch.zeros(face_adj.shape)
    curved_faces[rows, active_faces[rows, shuffled_indices]] = 1
    flat_faces = face_adj - curved_faces

    return flat_faces, curved_faces


def base_edge_parameters(node_euclidean_coords: torch.Tensor, edge_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the base edge parameters for the given samples.

    node_euclidean_coords: `torch.Tensor`
        A `(N,R//3,3)` tensor with the Euclidean coordinates
        of each node, ordered in the second dimension by node ID.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.

    edge_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.

    Returns: `tuple[torch.Tensor]`
        A `(N,R,-1)` tensor with the samples' corresponding
        flat edge parameters.

        A `(N,R,-1)` tensor with the samples' corresponding
        curved edge parameters.

        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
        The edge parameters of non-existent edges are 0.
    """

    # Stores the number of metamaterial samples
    num_samples = node_euclidean_coords.shape[0]

    # Stores the general shape of the edge parameter arrays
    edge_params_shape = (num_samples, EDGE_ADJ_SIZE, EDGE_BEZIER_COORDS)

    # Computes the flat/curved edge parameters
    flat_edge_params = flat_edge_parameters(node_euclidean_coords, edge_adj).reshape(edge_params_shape)
    curved_edge_params = (torch.rand(edge_params_shape)*2-1) * edge_adj.unsqueeze(-1)

    return flat_edge_params, curved_edge_params


def find_flat_and_curved_face_edges(num_nodes: int, edge_adj: torch.Tensor, flat_faces: torch.Tensor, curved_faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the flat/curved edge adjacencies that correspond to edges
    tied to an active face. Deals with edges that are tied to both
    an active flat face and curved face randomly.

    num_nodes: `int`
        The number of active nodes to use in every sample.

    edge_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.

    flat_faces: torch.Tensor
        A `(N,R)` tensor with the samples' face adjacencies
        just for the chosen flat faces.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.
        
    curved_faces: torch.Tensor
        A `(N,R)` tensor with the samples' face adjacencies
        just for the chosen curved faces.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the samples' edge adjacencies
        that correspond to edges tied to active flat faces.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.

        A `(N,R)` tensor with the samples' edge adjacencies
        that correspond to edges tied to active curved faces.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
    """

    # Stores the template
    flat_edges: torch.Tensor = torch.zeros(edge_adj.shape)
    curved_edges: torch.Tensor = torch.zeros(edge_adj.shape)

    # Runs through each face
    for n1 in range(num_nodes):
        for n2 in range(n1+1, num_nodes):
            for n3 in range(n2+1, num_nodes):

                # Stores the face index
                face_index = face_adj_index(n1,n2,n3)

                # Stores the edge_indices
                edge1_index = edge_adj_index(n1,n2)
                edge2_index = edge_adj_index(n1,n3)
                edge3_index = edge_adj_index(n2,n3)

                # Labels the edge as a flat face-edge
                face_val = flat_faces[:,face_index]
                flat_edges[:,edge1_index] += face_val
                flat_edges[:,edge2_index] += face_val
                flat_edges[:,edge3_index] += face_val
                
                # Labels the edge as a curved face-edge
                face_val = curved_faces[:,face_index]
                curved_edges[:,edge1_index] += face_val
                curved_edges[:,edge2_index] += face_val
                curved_edges[:,edge3_index] += face_val

    # Normalizes the edge adjacencies
    flat_edges = (flat_edges > 0).to(torch.float32)
    curved_edges = (curved_edges > 0).to(torch.float32)

    # Randomly deals with intersections
    intersections = torch.logical_and(flat_edges, curved_edges).to(torch.float32)
    keep_flat_edges = (torch.rand(intersections.shape) < 0.5).to(torch.float32)

    # Removes intersections
    flat_edges -= intersections * (1-keep_flat_edges)
    curved_edges -= intersections * keep_flat_edges

    return flat_edges, curved_edges

    
def choose_flat_and_curved_edges(num_nodes: int, num_curved_edges: int, edge_adj: torch.Tensor, flat_faces: torch.Tensor, curved_faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chooses which edges will correspond to flat/curved edges
    in the given samples.

    num_nodes: `int`
        The number of active nodes to use in every sample.

    num_curved_edges: `int`
        The number of curved edges to use in each sample.

    edge_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.

    flat_faces: torch.Tensor
        A `(N,R)` tensor with the samples' face adjacencies
        just for the chosen flat faces.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.
        
    curved_faces: torch.Tensor
        A `(N,R)` tensor with the samples' face adjacencies
        just for the chosen curved faces.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.

    Returns: `tuple[torch.Tensor]`
        A `(N,R)` tensor with the samples' edge adjacencies
        just for the chosen flat edges.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
        
        A `(N,R)` tensor with the samples' edge adjacencies
        just for the chosen curved edges.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
    """

    # Stores relevant values for the function
    num_samples = edge_adj.shape[0]

    # Stores the indices of active non-face/face edges
    flat_face_edges, curved_face_edges = find_flat_and_curved_face_edges(num_nodes, edge_adj, flat_faces, curved_faces)
    non_face_edges = (edge_adj - (flat_face_edges + curved_face_edges) > 0).to(torch.float32)
    num_edges = int(non_face_edges[0].sum().item())

    # Stores the indices of active non-face edges
    active_edges = torch.nonzero(non_face_edges)[:,1].reshape((num_samples, num_edges))

    # Generates shuffled indices
    shuffled_indices = torch.stack([torch.randperm(num_edges)[:num_curved_edges] for _ in range(num_samples)])
    rows = torch.arange(num_samples).view(-1, 1).expand(num_samples, num_curved_edges)

    # Chooses the non-face flat/curved edges
    curved_edges: torch.Tensor = torch.zeros(edge_adj.shape)
    curved_edges[rows, active_edges[rows, shuffled_indices]] = 1
    flat_edges: torch.Tensor = non_face_edges - curved_edges

    return flat_edges + flat_face_edges, curved_edges + curved_face_edges


def generate_edge_and_face_parameters(
        num_nodes: int, num_curved_edges: int, num_curved_faces: int, node_euclidean_coords: torch.Tensor,
        edge_adj: torch.Tensor, face_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random edge and face parameters according
    to the given sample specifications.

    num_nodes: `int`
        The number of active nodes to use in every sample.

    num_curved_edges: `torch.Tensor`
        The number of non-face curved edges to use in each sample.

    num_curved_edges: `torch.Tensor`
        The number of curved faces to use in each sample.

    node_euclidean_coords: `torch.Tensor`
        A `(N,R//3,3)` tensor with the Euclidean coordinates
        of each node, ordered in the second dimension by node ID.
        `N` is the number of samples.
        `R` is the size of the node position array in the
        Metamaterial representation.

    edge_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' edge adjacencies.
        `N` is the number of samples.
        `R` is the size of the edge adjacency array in the
        Metamaterial representation.
        
    face_adj: `torch.Tensor`
        A `(N,R)` tensor with the samples' face adjacencies.
        `N` is the number of samples.
        `R` is the size of the face adjacency array in the
        Metamaterial representation.

    Returns: `tuple[torch.Tensor, torch.Tensor]`
        A `(N,R)` tensor with the generated edge parameters.
        Contains only edge parameters for the given active edges.
        All other edge parameters are 0.
        `N` is the number of samples.
        `R` is the size of the edge parameters array in the
        Metamaterial representation.
        
        A `(N,R)` tensor with the generated face parameters.
        Contains only face parameters for the given active faces.
        All other face parameters are 0.
        `N` is the number of samples.
        `R` is the size of the face parameters array in the
        Metamaterial representation.
    """

    # Computes the base face parameters
    flat_face_params, curved_face_params = base_face_parameters(node_euclidean_coords, face_adj)

    # Stores which faces are to be flat/curved
    flat_faces, curved_faces = choose_flat_and_curved_faces(num_curved_faces, face_adj)

    # Computes the face parameters by combining only the target flat/curved face parameters
    face_params = flat_face_params * (flat_faces.unsqueeze(-1)) + curved_face_params * (curved_faces.unsqueeze(-1))

    # Computes the base edge parameters
    flat_edge_params, curved_edge_params = base_edge_parameters(node_euclidean_coords, edge_adj)

    # Stores which edges are to be flat/curved
    flat_edges, curved_edges = choose_flat_and_curved_edges(num_nodes, num_curved_edges, edge_adj, flat_faces, curved_faces)

    # Computes the edge parameters by combining only the target flat/curved edge parameters
    edge_params = flat_edge_params * (flat_edges.unsqueeze(-1)) + curved_edge_params * (curved_edges.unsqueeze(-1))

    # Resizes the parameters to match the requirement for the metamaterial representation
    num_samples = edge_params.shape[0]
    edge_params = edge_params.reshape((num_samples, -1))
    face_params = face_params.reshape((num_samples, -1))

    return edge_params, face_params


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
    edge_params = flat_edge_parameters(node_coords, edge_adj)

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
        The exact number of faces to use in each sample.
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
    edge_params = flat_edge_parameters(node_coords, edge_adj)

    # Stores the face parameters
    face_params = flat_face_parameters(node_coords, face_adj)

    # Stores the other parameters
    global_params = torch.ones((num_samples,1)) * 0.5 # Thickness

    return torch.cat([node_pos, edge_adj, edge_params, face_adj, face_params, global_params], dim=1)


def random_metamaterials(num_samples: int, num_nodes: int, num_edges: int, num_curved_edges: int, num_faces: int, num_curved_faces: int) -> torch.Tensor:
    """
    Generates random metamaterials based on the given attributes.
    The metamaterials satisfy the following properties:

    1) There are no disconnected components.
    2) Every face in the unit cube has at least one node.
    3) All nodes are on at least one face of the unit cube.
    4) All nodes are on at most three faces of the unit cube.

    num_samples: `int`
        The number of random metamaterial samples to generate.

    num_nodes: `int`
        The number of active nodes to use in every sample.

    num_edges: `torch.Tensor`
        The total number of non-face edges to use in each sample.

    num_curved_edges: `torch.Tensor`
        The number of non-face curved edges to use in each sample.

    num_faces: `torch.Tensor`
        The total number of faces to use in each sample.

    num_curved_edges: `torch.Tensor`
        The number of curved faces to use in each sample.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the random samples of metamaterials.
        `N` is the number of samples.
        `R` is the representation size.
    """

    assert num_nodes <= 1 + num_faces*2 + num_edges, "There are not enough edges and faces to make a fully connected metamaterial."

    # Stores the node positions
    node_pos, node_euclidean_coords = random_node_positions(num_samples, num_nodes)

    # Computes the edge/face adjacencies
    edge_adj, face_adj = generate_edge_and_face_adjacencies(num_samples, num_nodes, num_edges, num_faces)

    # Computes the edge/face parameters
    edge_params, face_params = generate_edge_and_face_parameters(num_nodes, num_curved_edges, num_curved_faces, node_euclidean_coords, edge_adj, face_adj)

    # Stores the other parameters
    global_params = torch.ones((num_samples,1)) * 0.5 # Thickness

    return torch.cat([node_pos, edge_adj, edge_params, face_adj, face_params, global_params], dim=1)

