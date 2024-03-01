import numpy as np
import matplotlib.pyplot as plt
from rep_utils import *

NODES_PER_FACE = 1
CUBE_FACES = 6
NUM_NODES = CUBE_FACES * NODES_PER_FACE + 1


def random_metamaterial(edge_prob=0.5, face_prob=0.5, with_faces=True):
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
    edge_adj = (np.random.rand(NUM_NODES * (NUM_NODES-1) // 2) < edge_prob).astype(float)

    # Generates the flattened face adjacency tensor (without redundancy)
    # Is effectively the flattened tensor equivalent of the upper triangle
    # of an adjacency matrix, but of the adjacency tensor.
    # The "upper tetrahedron" is a contiguous section of the tensor where no
    # index repeats in any axis.
    if with_faces:
        face_adj = (np.random.rand(NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6) < face_prob).astype(float)
    else:
        face_adj = np.zeros(NUM_NODES * (NUM_NODES-1) * (NUM_NODES-2) // 6)

    # Ensures the representation is of a validly constructed metamaterial
    face_adj = remove_invalid_faces(edge_adj, face_adj) # Removes faces without all edges in the rep
    edge_adj = remove_invalid_edges(node_pos, edge_adj, face_adj) # Removes edges intersecting with faces
    face_adj = remove_invalid_faces(edge_adj, face_adj) # Removes faces without all edges in the rep after edge removal

    return node_pos, edge_adj, face_adj


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
