import numpy as np

# User-controlled properties
NUM_NODES = 12 + 1 # Non-center nodes plus the single center node
EDGE_BEZIER_POINTS = 2 # The number of points to describe curved edges
EDGE_SEGMENTS = 32 # The number of segments to use to mesh edges/faces
FACE_BEZIER_POINTS = 1 # The number of points to described curved faces

# Automatically-chosen properties
NODE_POS_SIZE = (NUM_NODES-1) * 2
EDGE_ADJ_SIZE = NUM_NODES * (NUM_NODES-1) // 2
EDGE_PARAMS_SIZE = EDGE_ADJ_SIZE * EDGE_BEZIER_POINTS * 3
FACE_ADJ_SIZE = NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6
FACE_PARAMS_SIZE = FACE_ADJ_SIZE * FACE_BEZIER_POINTS
REP_SIZE = NODE_POS_SIZE + EDGE_ADJ_SIZE + EDGE_PARAMS_SIZE + FACE_ADJ_SIZE + FACE_PARAMS_SIZE


def euclidian_to_spherical(x: float, y: float, z: float):
    """
    Converts the given x,y,z Euclidian triplet into spherical coordinates.

    x: float
        The x-coordinate to be converted.

    y: float
        The y-coordinate to be converted.

    z: float
        The z-coordinate to be converted.

    Returns: ndarray
        The first entry is the angle off the z-axis, normalized to [0,1].
        The second entry is the angle on the xy plane, normalized to [0,1].
        If the input coordinate is too close to (0,0,0), returns (0,0).
    """

    # Computes the radial distance from the center
    radius = np.sqrt(x**2 + y**2 + z**2)

    # Checks for a non-collapsed radius
    if radius < 1e-4:
        return np.zeros(2)
    
    # Computes the angles in spherical coordinates
    theta = np.arccos(z/radius)
    if np.abs(np.sin(theta)) < 1e-4:
        phi = 0
    else:
        phi = np.arctan2(y/np.sin(theta)/radius, x/np.sin(theta)/radius)
        phi = phi if phi >= 0 else phi+2*np.pi

    return np.array([theta/np.pi, phi/(2*np.pi)])


def spherical_to_euclidian(theta, phi, radius=None):
    """
    Converts the given spherical coordinates into Euclidian coordinates.

    theta: float or ndarray
        The angle off the z-axis, [0, pi].

    phi: float or ndarray
        The angle on the xy plane, [0, 2pi].

    radius: float or ndarray or None
        The radius from the spherical center. If None, then
        assumed to be 1.
 
    Returns: ndarray
        The first entry is the x-coordinate of the point.
        The second entry is the y-coordinate of the point.
        The third entry is the z-coordinate of the point.
    """

    # Fixes the radius if needed
    if radius is None:
        if type(theta) == np.ndarray:
            radius = np.ones(theta.shape)
        else:
            radius = 1.

    # Computes the Euclidian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.array([x,y,z])


def project_onto_cube(x: float, y: float, z: float, grid_lines=0, bias_cutoff=1):
    """
    Projects the given coordinates onto the surface of the unit cube
    centered at (0.5, 0.5, 0.5).

    x: float
        The x-coordinate to project.

    y: float
        The y-coordinate to project.

    z: float
        The z-coordinate to project.

    grid_lines: int
        How many grid lines to bias the points toward. If is 0, no biasing
        will occur. Must be at least 2 for biasing.

    bias_cutoff: float
        The cutoff of how proportionally far from the grid center a node must
        be before it is biased.

    Returns: ndarray
        The points projected onto the surface of the unit cube.
    """

    # Projects onto the cube
    pos = np.array([x,y,z])
    pos /= np.max(np.abs(pos))
    pos = (pos + np.ones(3)) / 2

    # Biases the points toward the grid lines
    if grid_lines:

        # Gets the distance of the node from the center of a grid square
        grid_lines -= 1
        dr = (pos*grid_lines) % 1
        
        # Computes whether the node is in the bias threshold
        cutoffs = np.abs(dr - 0.5) <= (0.5 * bias_cutoff)

        # Computes the biased position of the node
        scaled_pos_int = (pos*grid_lines)//1
        normalized_grid_dist = (dr - (1-bias_cutoff)/2)/bias_cutoff
        pos = cutoffs * (scaled_pos_int + normalized_grid_dist) + (1-cutoffs) * np.round(pos * grid_lines, decimals=0)
        pos /= grid_lines

    return pos


def edge_adj_index(node1: int, node2: int):
    """
    Computes the index at which the two nodes' edge adjacency is contained
    in the edge adjacency representation array.

    node1: int
        The ID of the first node. Must not be equal to node2. Must be < NUM_NODES.

    node2: int
        The ID of the second node. Must not be equal to node1. Must be < NUM_NODES.

    Returns: int
        The index at which the given nodes' edge adjacency is located in the
        edge adjacency array.
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


def to_edge_adj_rep(edge_adj_matrix: np.ndarray):
    """
    Converts the given edge adjacency matrix into the edge adjacency
    representation array.

    edge_adj_matrix: ndarray
        A matrix where the entry at index (i,j) is 1 if the nodes with
        IDs i and j have an edge between them, or 0 otherwise.

    Returns: ndarray
        The edge adjacency array as described in the Metamaterial initializer.
    """

    # Prepares the rep array
    edge_adj = np.zeros(EDGE_ADJ_SIZE).astype(float)

    # Stores each non-redundant edge adjacency from the matrix
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            edge_adj[edge_adj_index(n1, n2)] = edge_adj_matrix[n1, n2]

    return edge_adj


def to_edge_adj_matrix(edge_adj: np.ndarray):
    """
    Converts the given edge adjacency representation into an edge adjacency
    matrix.

    edge_adj: ndarray
        The edge adjacency array as described in the Metamaterial initializer.

    Returns: ndarray
        A matrix where the entry at index (i,j) is 1 if the nodes with
        IDs i and j have an edge between them, or 0 otherwise.
    """

    # Prepares the adjacency matrix
    edge_adj_matrix = np.zeros((NUM_NODES,) * 2)

    # Stores each edge adjacency from the rep
    for n1 in range(NUM_NODES):
        for n2 in range(NUM_NODES):

            # Gets the edge adjacency from the rep
            if n1 != n2:
                edge_adj_matrix[n1, n2] = edge_adj[edge_adj_index(n1, n2)]

    return edge_adj_matrix


def face_adj_index(node1: int, node2: int, node3: int):
    """
    Computes the index at which the three nodes' face adjacency is contained
    in the face adjacency representation array.

    node1: int
        The ID of the first node. Must not be equal to node2 or node3.
        Must be < NUM_NODES.

    node2: int
        The ID of the second node. Must not be equal to node2 or node3.
        Must be < NUM_NODES.

    node3: int
        The ID of the third node. Must not be equal to node2 or node3.
        Must be < NUM_NODES.

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


def to_face_adj_rep(face_adj_tensor: np.ndarray):
    """
    Converts the given face adjacency tensor into the face adjacency
    representation array.

    face_adj_tensor: ndarray
        A 3D tensor where the entry at index (i,j,k) is 1 if the nodes with
        IDs i, j, and k have a triangular face between them, or 0 otherwise.

    Returns: ndarray
        The face adjacency array as described in the Metamaterial initializer.
    """

    # Prepares the rep array
    face_adj = np.zeros(FACE_ADJ_SIZE).astype(float)

    # Stores each non-redundant face adjacency from the tensor
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):
                face_adj[face_adj_index(n1, n2, n3)] = face_adj_tensor[n1, n2, n3]

    return face_adj


def to_face_adj_tensor(face_adj: np.ndarray):
    """
    Converts the given face adjacency representation into a face adjacency
    tensor.

    face_adj: ndarray
        The face adjacency array as described in the Metamaterial initializer.

    Returns: ndarray
        A 3D tensor where the entry at index (i,j,k) is 1 if the nodes with
        IDs i, j, and k have a triangular face between them, or 0 otherwise.
    """

    # Prepares the adjacency tensor
    face_adj_matrix = np.zeros((NUM_NODES,) * 3)

    # Stores each face adjacency from the rep
    for n1 in range(NUM_NODES):
        for n2 in range(NUM_NODES):
            for n3 in range(NUM_NODES):

                # Gets the face adjacency from the rep
                if not (n1 == n2 or n1 == n3 or n2 == n3):
                    face_adj_matrix[n1, n2, n3] = face_adj[face_adj_index(n1, n2, n3)]

    return face_adj_matrix


# Computes the binomial coefficients for a Bezier curve
binomial_coefficients = np.ones((1,EDGE_BEZIER_POINTS+2))
for i in range(1,EDGE_BEZIER_POINTS+1):
    binomial_coefficients[0,i] = binomial_coefficients[0,i-1] * (EDGE_BEZIER_POINTS+2-i) // i

# Computes the general coefficients for a Bezier curve
BEZIER_CURVE_COEFFICIENTS = binomial_coefficients * np.array([[
    (t/EDGE_SEGMENTS) ** i * (1-t/EDGE_SEGMENTS) ** (EDGE_BEZIER_POINTS+1-i)
        for i in range(EDGE_BEZIER_POINTS+2)]
            for t in range(EDGE_SEGMENTS+1)
])


def find_edge_params(target_edge_points: np.ndarray) -> np.ndarray:
    """
    Runs regression to find the edge parameters that most closely
    produce the given target edge points.

    target_edge_points: np.ndarray
        A 2D numpy array of the edge points to try to most closely
        match. The first axis should separate the target edge points
        (in order from t=0 to t=1). The second axis should separate
        the coordinates of a given edge point. There must be
        `EDGE_SEGMENTS+1` edge points, as it must include the node
        positions at the ends.

    Returns: np.ndarray
        The edge parameters that most closely match the given edge
        points, formatted in the way that the Metamaterial class
        edge_params value should be for a particular edge.
    """

    # Separates the node positions (which are fixed)
    node1_pos = target_edge_points[:1,:]
    node2_pos = target_edge_points[-1:,:]

    # Computes the effect of the node positions
    node1_effect = BEZIER_CURVE_COEFFICIENTS[1:-1,:1] @ node1_pos
    node2_effect = BEZIER_CURVE_COEFFICIENTS[1:-1,-1:] @ node2_pos

    # Computes the target output (the coordinates vary on second axis)
    b = target_edge_points[1:-1,:]
    b -= node1_effect + node2_effect

    # Computes the linear system's matrix
    A = BEZIER_CURVE_COEFFICIENTS[1:-1,1:-1]

    # Solves the system for each coordinate
    edge_params = np.concatenate([np.linalg.solve(A.T @ A, A.T @ b[:,i]).reshape(EDGE_BEZIER_POINTS, 1) for i in range(3)], axis=1)
    return edge_params.flatten()

