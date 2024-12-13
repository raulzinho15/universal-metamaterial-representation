import torch
import numpy as np
import time
import random
from representation.rep_utils import *


def random_trusses(num_samples: int, num_nodes: int, edge_prob: float):
    """
    Generates random truss metamaterials based on the given attributes.
    The metamaterials are generated such that:
        1) It has no disconnected components.
        2) Every face in the unit cube has at least one node. (IMPLEMENTED WHEN >=6 NODES)
        3) The material has the number of active nodes specified.
        4) The material is a truss (i.e., has only flat edges).
        5) All nodes are on the face of the unit cube.

    num_samples: `int`
        The number of random truss samples to generate.

    num_nodes: `int`
        The exact number of nodes to use in every sample.
        Must be at least 2.

    edge_prob: `float`
        The probability of an edge existing.

    Returns: `torch.Tensor`
        A `(N,R)` tensor with the random samples of truss
        metamaterials.
        `N` is the number of samples.
        `R` is the representation size.
    """

    # Checks for valid parameters
    assert num_nodes > 1, "No edges can be made with less than 2 nodes."

    # Computes the mask for edges to exclude
    # edges beyond those on the specified nodes
    edge_mask = torch.tensor([[
        1 if n1 < num_nodes and n2 < num_nodes else 0
            for n1 in range(NUM_NODES)
                for n2 in range(n1+1, NUM_NODES)
    ]])


    ### NODE POSITIONS

    # Initializes the random node positions
    node_coords = torch.rand((num_samples,NODE_POS_SIZE//3,3))

    # Sets unused nodes to (0,0,0) in pseudo-spherical coordinates
    node_coords[:,num_nodes:] = 0.5

    # Ensures that each face has at least one node
    node_coords[:,[0,1,2],[0,1,2]] = 0 # x,y,z = 0 (ASSUMES 6 NODES!!!!!!!!!!)
    node_coords[:,[3,4,5],[0,1,2]] = 1 # x,y,z = 1 (ASSUMES 6 NODES!!!!!!!!!!)

    # Stores the pseudo-spherical coordinates
    node_pos = euclidean_to_pseudo_spherical_torch(node_coords).reshape((num_samples,-1))
    

    ### EDGE ADJACENCIES

    # Initializes the random edge adjacencies
    edge_adj = (torch.rand((num_samples,EDGE_ADJ_SIZE)) < edge_prob) * edge_mask

    # Stores the edge indices for each node
    nodes_to_edge = torch.tensor([[edge_adj_index(n1,n2) for n2 in range(num_nodes)] for n1 in range(num_nodes)])

    # Ensures each node is included in one connected component
    node_perms = torch.stack([torch.randperm(num_nodes) for _ in range(num_samples)])
    base_indices = [i for i in range(num_samples)]
    for n in range(1,num_nodes):
        this_node = node_perms[:,n]
        other_node = node_perms[base_indices,torch.randint(0,n,(num_samples,))]
        edge_index = nodes_to_edge[this_node,other_node]
        edge_adj[base_indices,edge_index] = 1


    ### EDGE PARAMETERS

    # Stores the node indices for each edge
    node_indices = torch.tensor([[n1, n2] for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES)])

    # Stores the vector describing each edge
    edge_vectors = node_coords[:,node_indices[:,1]] - node_coords[:,node_indices[:,0]]

    # Stores the edge parameters that make a straight edge
    edge_params = torch.cat([edge_vectors/3, 2*edge_vectors/3], dim=2) * edge_adj.unsqueeze(-1)
    edge_params = edge_params.reshape((num_samples,-1))


    ### OTHER PARAMETERS
    face_adj = torch.zeros((num_samples,FACE_ADJ_SIZE))
    face_params = torch.zeros((num_samples,FACE_PARAMS_SIZE))
    global_params = torch.ones((num_samples,1)) * 0.5 # Thickness

    return torch.cat([node_pos, edge_adj, edge_params, face_adj, face_params, global_params], dim=1)
