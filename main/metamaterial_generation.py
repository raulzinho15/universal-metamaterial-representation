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
        corresponding unit cube face. The nodes appear in sequential order. The
        logic for extracting nodes' positions can be found in get_node_x/y/z().
        
        The second entry in this tuple is a 1d numpy array edge adjacency
        array where a 1 at the corresponding adjacency matrix's i-th row and
        j-th column means that nodes i and j are connected by an edge. All other
        entries are 0. The logic for extracting the edge adjacencies can be found
        in the have_edge() function.
        
        The third entry in this tuple is a 1d numpy array face adjacency
        array where a 1 at the corresponding adjacency tensor's (i,j,k) index
        means that nodes i, j, and k are connected by a triangular face. All other
        entries are 0. The logic for extracting the face adjacencies can be found
        in the have_face() function.
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


def get_node_x(node, node_pos):
    """
    Computes what the x coordinate of the node with the given ID is
    based on the given node positions.

    node: int
        The ID of the node.

    node_pos: ndarray
        The node positions as described in the specification of
        the random_metamaterial() function.

    Returns: float
        The x coordinate of the node.
    """

    # Needs to be adapted if node_pos rep changes
    return 0.5 if node == NUM_NODES-1 else node_pos[node*2] if node//2 != 2 else node%2


def get_node_y(node, node_pos):
    """
    Computes what the y coordinate of the node with the given ID is
    based on the given node positions.

    node: int
        The ID of the node.

    node_pos: ndarray
        The node positions as described in the specification of
        the random_metamaterial() function.

    Returns: float
        The y coordinate of the node.
    """

    # Needs to be adapted if node_pos rep changes
    return 0.5 if node == NUM_NODES-1 else node_pos[node*2] if node//2 != 1 else node%2


def get_node_z(node, node_pos):
    """
    Computes what the z coordinate of the node with the given ID is
    based on the given node positions.

    node: int
        The ID of the node.

    node_pos: ndarray
        The node positions as described in the specification of
        the random_metamaterial() function.

    Returns: float
        The z coordinate of the node.
    """

    # Needs to be adapted if node_pos rep changes
    return 0.5 if node == NUM_NODES-1 else node_pos[node*2] if node//2 != 0 else node%2


def edge_adj_index(node1, node2):
    """
    Computes the index at which the two nodes' edge adjacency is contained
    in the edge adjacency representation returned by random_metamaterial().

    node1: int
        The ID of the first node.

    node2: int
        The ID of the second node.

    Returns: int
        The index at which the given nodes' edge adjacency is located.
    """

    # Sorts the nodes by ascending index size
    node1, node2 = sorted((node1, node2))

    # Computes the index at which the edge adjacency is in the flattened vector
    # start2d = NUM_NODES * (NUM_NODES-1) // 2 - (NUM_NODES-node1) * (NUM_NODES-node1-1) // 2
    start2d = node1 * (2*NUM_NODES - node1 - 1) // 2 # Needs to be changed if rep changes
    start1d = node2 - node1 - 1

    return start2d + start1d


def have_edge(node1, node2, edge_adj):
    """
    Checks whether the two given nodes have an edge between them
    based on the given edge adjacencies.

    node1: int
        The ID of the first node.

    node2: int
        The ID of the second node.

    edge_adj: ndarray
        The edge adjacencies as described in the specification of
        the random_metamaterial() function.

    Returns: bool
        Whether there is an edge between the two given nodes.
    """

    # Trivially excludes an edge from identical nodes
    if node1 == node2:
        return False

    return edge_adj[edge_adj_index(node1, node2)] == 1


def to_edge_adj_matrix(edge_adj):
    """
    Converts the given edge adjacencies into an edge adjacency matrix.

    edge_adj: ndarray
        The edge adjacencies as described in the specification of
        the random_metamaterial() function.

    Returns: ndarray
        An edge adjacency matrix, where the position (i,j) is 1 if and
        only if there is an edge connecting the i-th and j-th nodes.
        All other entries are 0.
    """
    return np.array([[have_edge(n1, n2, edge_adj)
                    for n1 in range(NUM_NODES)]
                        for n2 in range(NUM_NODES)]).astype(int)


def face_adj_index(node1, node2, node3):
    """
    Computes the index at which the three nodes' face adjacency is contained
    in the face adjacency representation returned by random_metamaterial().

    node1: int
        The ID of the first node.

    node2: int
        The ID of the second node.

    node3: int
        The ID of the third node.

    Returns: int
        The index at which the given nodes' face adjacency is located.
    """

    # Sorts the nodes by ascending index size
    node1, node2, node3 = sorted((node1, node2, node3))

    # Computes the index at which the relevant window of face adjacencies is located
    start3d = NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6 - (NUM_NODES-node1) * (NUM_NODES-node1-1) * (NUM_NODES-node1-2) // 6
    start2d = (node2-node1-1) * (2*NUM_NODES - node2 + node1 - 2) // 2 # Needs to be changed if rep changes
    start1d = node3 - node2 - 1

    return start3d + start2d + start1d


def have_face(node1, node2, node3, face_adj):
    """
    Checks whether the three given nodes have a face between them
    based on the given face adjacencies.

    node1: int
        The ID of the first node.

    node2: int
        The ID of the second node.

    node3: int
        The ID of the third node.

    face_adj: ndarray
        The face adjacencies as described in the specification of
        the random_metamaterial() function.

    Returns: bool
        Whether there is a face between the three given nodes.
    """

    # Trivially excludes a face from identical nodes
    if node1 == node2 or node2 == node3 or node1 == node3:
        return False

    return face_adj[face_adj_index(node1, node2, node3)] == 1

def to_face_adj_tensor(face_adj):
    """
    Converts the given face adjacencies into an face adjacency tensor.

    face_adj: ndarray
        The face adjacencies as described in the specification of
        the random_metamaterial() function.

    Returns: ndarray
        A face adjacency tensor, where the position (i,j,k) is 1 if and
        only if there is a face connecting the i-th, j-th, and k-th
        nodes. All other entries are 0.
    """
    return np.array([[[have_face(n1, n2, n3, face_adj)
                    for n1 in range(NUM_NODES)]
                        for n2 in range(NUM_NODES)]
                            for n3 in range(NUM_NODES)]).astype(int)
