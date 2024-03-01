import numpy as np
import utils

CUBE_FACES = 6
NODES_PER_FACE = 2
CUBE_EDGES = 12
NODES_PER_EDGE = 2
CUBE_VERTICES = 8
NODES_PER_VERTEX = 1
NUM_NODES = (CUBE_FACES * NODES_PER_FACE            # Cube face nodes
            + CUBE_EDGES * NODES_PER_EDGE           # Cube edge nodes
            + CUBE_VERTICES * NODES_PER_VERTEX + 1) # Cube vertex/center nodes


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

    # Cube center node
    if node == NUM_NODES-1:
        return 0.5
    
    # Cube face node
    if node < NODES_PER_FACE * CUBE_FACES:
        if node // (2*NODES_PER_FACE) != 2: # Not in a face parallel to the yz plane
            return node_pos[node*2]
        return (node//NODES_PER_FACE) % 2 # In a face parallel to the yz plane
    
    # Cube edge node
    if node < NODES_PER_FACE * CUBE_FACES + NODES_PER_EDGE * CUBE_EDGES:
        node -= NODES_PER_FACE * CUBE_FACES # Offsets the node 
        if node // (4*NODES_PER_EDGE) == 2: # In an edge that runs along the x axis
            return node_pos[NODES_PER_FACE * CUBE_FACES * 2 + node]
        return (node // (2*NODES_PER_EDGE)) % 2 # In an edge that does not run along the x axis
    
    # Cube vertex node
    node -= NODES_PER_FACE * CUBE_FACES + NODES_PER_EDGE * CUBE_EDGES
    return node//4


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

    # Cube center node
    if node == NUM_NODES-1:
        return 0.5
    
    # Cube face node
    if node < NODES_PER_FACE * CUBE_FACES:
        if node // (2*NODES_PER_FACE) != 1: # Not in a face parallel to the xz plane
            return node_pos[node*2 + (1 if node // (2*NODES_PER_FACE) == 0 else 0)]
        return (node//NODES_PER_FACE) % 2 # In a face parallel to the xz plane
    
    # Cube edge node
    if node < NODES_PER_FACE * CUBE_FACES + NODES_PER_EDGE * CUBE_EDGES:
        node -= NODES_PER_FACE * CUBE_FACES # Offsets the node 
        if node // (4*NODES_PER_EDGE) == 1: # In an edge that runs along the y axis
            return node_pos[NODES_PER_FACE * CUBE_FACES * 2 + node]
        return (node // ((2 if node // (4*NODES_PER_EDGE) == 2 else 1) * NODES_PER_EDGE)) % 2 # In an edge that does not run along the y axis
    
    # Cube vertex node
    node -= NODES_PER_FACE * CUBE_FACES + NODES_PER_EDGE * CUBE_EDGES
    return (node//2) % 2


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

    # Cube center node
    if node == NUM_NODES-1:
        return 0.5
    
    # Cube face node
    if node < NODES_PER_FACE * CUBE_FACES:
        if node // (2*NODES_PER_FACE) != 0: # Not in a face parallel to the xy plane
            return node_pos[node*2+1]
        return (node//NODES_PER_FACE) % 2 # In a face parallel to the xy plane
    
    # Cube edge node
    if node < NODES_PER_FACE * CUBE_FACES + NODES_PER_EDGE * CUBE_EDGES:
        node -= NODES_PER_FACE * CUBE_FACES # Offsets the node 
        if node // (4*NODES_PER_EDGE) == 0: # In an edge that runs along the z axis
            return node_pos[NODES_PER_FACE * CUBE_FACES * 2 + node]
        return (node // NODES_PER_EDGE) % 2 # In an edge that does not run along the z axis
    
    # Cube vertex node
    node -= NODES_PER_FACE * CUBE_FACES + NODES_PER_EDGE * CUBE_EDGES
    return node%2


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


def remove_invalid_edges(node_pos, edge_adj, face_adj):
    """
    Removes all the edges described in the edge adjacency array which
    intersect with faces described in the face adjacency array. Does
    not mutate the original array.

    node_pos: ndarray
        The node positions as described in the specification of the
        random_metamaterial() function.

    edge_adj: ndarray
        The edge adjacencies as described in the specification of
        the random_metamaterial() function.

    face_adj: ndarray
        The face adjacencies as described in the specification of
        the random_metamaterial() function.

    Returns: ndarray
        A new edge adjacency array such that no edges and faces in
        the metamaterial intersect.
    """
    edge_adj = np.copy(edge_adj)    

    # Runs through each possible face
    for nf1 in range(NUM_NODES):
        for nf2 in range(nf1+1, NUM_NODES):
            for nf3 in range(nf2+1, NUM_NODES):

                # Skips non-faces
                if not face_adj[face_adj_index(nf1, nf2, nf3)]:
                    continue

                # Runs through each possible edge
                face_nodes = (nf1, nf2, nf3)
                for ne1 in range(NUM_NODES):

                    # Skips node on the face
                    if ne1 in face_nodes:
                        continue

                    for ne2 in range(ne1+1, NUM_NODES):

                        # Skips node on the face
                        if ne2 in face_nodes:
                            continue

                        # Skips non-edges
                        index = edge_adj_index(ne1, ne2)
                        if not edge_adj[index]:
                            continue

                        # Checks for intersection
                        positions = [np.array([get_node_x(n, node_pos), get_node_y(n, node_pos), get_node_z(n, node_pos)])
                                        for n in (nf1, nf2, nf3, ne1, ne2)]
                        if utils.triangle_line_intersection(*positions):
                            edge_adj[index] = 0
    return edge_adj
