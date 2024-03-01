import numpy as np

NODES_PER_FACE = 1
CUBE_FACES = 6
NUM_NODES = CUBE_FACES * NODES_PER_FACE + 1

def random_metamaterial():
    """
    Generates a random metamaterial's representation with its node positions,
    edge relations, and face relations. The 7th node at the center of the cube
    is not explicitly included in the node position array since it is constant
    across all representations.

    Returns: tuple of ndarrays
        The first entry in this tuple is a 1d numpy array where every row pair
        of values is the normalized 2D position of its corresponding node on its
        corresponding unit cube face. The nodes appear in sequential order.
        
        The second entry in this tuple is a 1d numpy array edge adjacency
        array where a 1 at the corresponding adjacency matrix's i-th row and
        j-th column means that nodes i and j are connected by an edge. All other
        entries are 0.
        
        The third entry in this tuple is a 1d numpy array face adjacency
        array where a 1 at the corresponding adjacency tensor's (i,j,k) index
        means that nodes i, j, and k are connected by a triangular face. All other
        entries are 0.
    """
    
    # Generates the flattened normalized [0,1) cube face node 2D positions.
    # The i-th pair of values correspond to the non-trivial
    # coordinates for the i-th node in the graph.
    node_pos = np.random.rand(NODES_PER_FACE * CUBE_FACES * 2)

    # Generates the flattened edge adjacency matrix (without redundancy).
    # Is effectively the flattened upper triangle of an adjacency matrix
    edge_adj = (np.random.rand(NUM_NODES * (NUM_NODES-1) // 2) < 0.5).astype(float)

    # Generates the flattened face adjacency tensor (without redundancy)
    # Is effectively the flattened tensor equivalent of the upper triangle
    # of an adjacency matrix, but of the adjacency tensor.
    # The "upper triangle" is a contiguous section of the tensor where no
    # index repeats in any axis.
    face_adj = (np.random.rand(NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6) < 0.5).astype(float)

    return node_pos, edge_adj, face_adj
