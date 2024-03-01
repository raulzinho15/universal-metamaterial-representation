import numpy as np
import matplotlib.pyplot as plt

NODES_PER_FACE = 1
CUBE_FACES = 6
NUM_NODES = CUBE_FACES * NODES_PER_FACE + 1

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
    offset2d = node1 * (2*NUM_NODES - node1 - 1) // 2
    offset1d = node2 - node1 - 1

    # The idea for offset2d comes from the fact that at the upper triangle, the
    # number of entries per row is a triangular number, starting from the bottom
    # row. Treating node1 as the row index, the parial sum of higher triangular
    # numbers can be computed as the offset2d. Intuitively, offset2d determines
    # which row of the upper triangle is being inspected. The formula used was
    # derived from simple algebraic manipulation of the original expression.

    # The idea for offset1d comes from the fact that as you move down the rows of
    # the upper triangle, the rows become shorter. Intuitively, offset1d
    # determines which column of the upper triangle is being inspected.

    return offset2d + offset1d


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
                        for n2 in range(NUM_NODES)]).astype(float)


def to_edge_adj_rep(edge_adj_matrix):
    """
    Converts the given edge adjacency matrix into the edge adjacency
    representation described in the specification of the
    random_metamaterial() function.
    """

    # Prepares the rep array
    edge_adj = np.zeros(NUM_NODES * (NUM_NODES-1) // 2).astype(float)

    # Stores each non-redundant edge adjacency from the matrix
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            edge_adj[edge_adj_index(n1, n2)] = edge_adj_matrix[n1, n2]

    return edge_adj


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
    offset3d = NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6 - (NUM_NODES-node1) * (NUM_NODES-node1-1) * (NUM_NODES-node1-2) // 6
    offset2d = (NUM_NODES-node1-1) * (NUM_NODES-node1-2) // 2 - (NUM_NODES-node1-1 - (node2-node1-1)) * (NUM_NODES-node1-1 - (node2-node1-1) - 1) // 2
    offset1d = node3 - node2 - 1

    # The idea for offset3d comes from very similar logic as that of offset2d.
    # Namely, the number of entries at each depth along the tensor is a
    # tetrahedral number, starting from the last depth. Intuitively, offset3d
    # determines which depth of the upper tetrahedron is being inspected. The
    # formula used was derived from simple algebraic manipulation of the original
    # expression.

    # The idea for offset2d comes from the fact that at an upper tetrahedron, the
    # number of entries per row is a triangular number, starting from the bottom
    # row. Treating node1 as the row index, the parial sum of higher triangular
    # numbers can be computed as the offset2d. Intuitively, offset2d determines
    # which row of the upper tetrahedron is being inspected. The formula used was
    # derived from simple algebraic manipulation of the original expression.

    # The idea for offset1d comes from the fact that as you move down the rows of
    # the upper tetrahedron, the rows become shorter. Intuitively, offset1d
    # determines which column of the upper tetrahedron is being inspected.

    return offset3d + offset2d + offset1d


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
                            for n3 in range(NUM_NODES)]).astype(float)


def to_face_adj_rep(face_adj_tensor):
    """
    Converts the given face adjacency tensor into the face adjacency
    representation described in the specification of the
    random_metamaterial() function.
    """

    # Prepares the rep array
    face_adj = np.zeros(NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6).astype(float)

    # Stores each non-redundant face adjacency from the tensor
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):
                # print(n1, n2, n3)
                face_adj[face_adj_index(n1, n2, n3)] = face_adj_tensor[n1, n2, n3]

    return face_adj


def random_metamaterial(with_faces=True):
    """
    Generates a random metamaterial's representation with its node positions,
    edge relations, and face relations. The node at the center of the cube
    is not explicitly included in the node position array since its position
    is constant across all metamaterials, namely, (0.5, 0.5, 0.5).

    Returns: tuple of ndarrays
        The first entry in this tuple is a 1d numpy array where every row pair
        of values is the normalized 2D position of its corresponding node on its
        corresponding unit cube face. The nodes appear in sequential order. The
        logic for extracting nodes' positions can be found in get_node_x/y/z().
        
        The second entry in this tuple is a 1d numpy array edge adjacency
        array, where a 1 at the corresponding adjacency matrix's i-th row and
        j-th column means that nodes i and j are connected by an edge. All other
        entries are 0. The logic for extracting the edge adjacencies can be found
        in the have_edge() function.
        
        The third entry in this tuple is a 1d numpy array face adjacency
        array, where a 1 at the corresponding adjacency tensor's (i,j,k) index
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
    # The "upper tetrahedron" is a contiguous section of the tensor where no
    # index repeats in any axis.
    if with_faces:
        face_adj = (np.random.rand(NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6) < 0.5).astype(float)
    else:
        face_adj = np.zeros(NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6)

    # Ensures the representation is of a non-self-intersecting metamaterial
    face_adj = remove_invalid_faces(edge_adj, face_adj)

    return node_pos, edge_adj, face_adj


def remove_invalid_faces(edge_adj, face_adj):
    """
    Removes all the faces described in the face adjacency array whose
    three edges are not encoded in the edge adjacency. Does not mutate
    the original array.

    edge_adj: ndarray
        The edge adjacencies as described in the specification of
        the random_metamaterial() function.

    face_adj: ndarray
        The face adjacencies as described in the specification of
        the random_metamaterial() function.

    Returns: ndarray
        A new face adjacency array such that all three of a face's edges
        are also encoded in edge_adj.
    """

    # Avoids mutation
    face_adj = np.copy(face_adj)

    # Checks if every face validly has its corresponding 3 edges
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):

                # Skips non-face edge triplets
                index = face_adj_index(n1, n2, n3)
                if not face_adj[index]:
                    continue

                # Removes the invalid face (i.e., a face with insufficient edges)
                if not (edge_adj[edge_adj_index(n1, n2)] and edge_adj[edge_adj_index(n1, n3)] and edge_adj[edge_adj_index(n2, n3)]):
                    face_adj[index] = 0

    return face_adj


def plot_metamaterial(filename, node_pos, edge_adj, face_adj):
    """
    Plots the metamaterial with the given representation at the given filename.

    filename: str
        The name of the file at which the plot image will be saved.

    node_pos: ndarray
        The node position array as described in the specification of the
        random_metamaterial() function.

    edge_adj: ndarray
        The edge adjacency array as described in the specification of the
        random_metamaterial() function.

    face_adj: ndarray
        The face adjacency array as described in the specification of the
        random_metamaterial() function.
    """
    
    # Sets up the 3d plot environment
    fig = plt.figure()
    plot3d = fig.add_subplot(projection="3d")
    plot3d.set_xlabel("x")
    plot3d.set_ylabel("y")
    plot3d.set_zlabel("z")

    # Plots each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips unconnected nodes
            if not have_edge(n1, n2, edge_adj):
                continue

            # Computes the edge coordinates
            x = [get_node_x(n1, node_pos), get_node_x(n2, node_pos)]
            y = [get_node_y(n1, node_pos), get_node_y(n2, node_pos)]
            z = [get_node_z(n1, node_pos), get_node_z(n2, node_pos)]

            plot3d.plot(x, y, zs=z, linewidth=5)

    # Plots each face
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):

                # Skips unconnected nodes
                if not have_face(n1, n2, n3, face_adj):
                    continue

                # Computes the face coordinates
                x = [get_node_x(n1, node_pos), get_node_x(n2, node_pos), get_node_x(n3, node_pos)]
                y = [get_node_y(n1, node_pos), get_node_y(n2, node_pos), get_node_y(n3, node_pos)]
                z = [get_node_z(n1, node_pos), get_node_z(n2, node_pos), get_node_z(n3, node_pos)]

                plot3d.plot_trisurf(x, y, z)

    plt.savefig(filename)

    # Plays a rotation animation
    for angle in range(0, 360*5):
        plot3d.view_init(30, angle)
        plt.draw()
        plt.pause(.002)
