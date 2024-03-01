import numpy as np
import matplotlib.pyplot as plt
from rep_utils import *


def random_metamaterial(edge_prob=0.5, face_prob=0.5, with_faces=True):
    """
    Generates a random metamaterial's representation with its node positions,
    edge relations, and face relations. Implicitly determinable node positions
    are not explicitly included in the node position array since they are
    constant across all metamaterials.

    Returns: tuple of ndarrays
        The first entry in this tuple is a 1d numpy array where every row pair
        or single value is the normalized position of its corresponding node on its
        corresponding unit cube face/edge. The nodes appear in the following
        sequential order.
        
        The first (NODES_PER_FACE * CUBE_FACES) 2D pairs correspond to the nodes at
        the cube's faces. There are the same number of nodes on each cube face. The
        order of cube faces in the array is z=0, z=1, y=0, y=1, x=0, x=1, where the
        coordinates in the node position array appear in the order (x,y,z), and
        exclude whichever dimension whose plane on which the node lies (hence 2D).
        Co-face nodes are next to each other in the array.

        The next (NODES_PER_EDGE * CUBE_EDGES) values correspond to the nodes at the
        cube's edges. There are the same number of nodes on each edge. The order of edges
        in the array is (x=0, y=0), (x=0, y=1), (x=1, y=0), (x=1, y=1), (x=0, z=0),
        (x=0, z=1), (x=1, z=0), (x=1, z=1), (y=0, z=0), (y=0, z=1), (y=1, z=0), (y=1, z=1),
        where the coordinate stored in this array is in the non-constant axis for that
        node. Co-edge nodes are next to each other in the array.

        The ordering of nodes in this array is the ordering that is logically used when
        using the edge/face adjacency representations. The logic for extracting nodes'
        positions can be found in the get_node_x/y/z() functions.
        
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
    node_pos = np.random.rand((
        NODES_PER_FACE * CUBE_FACES * 2 # Cube face node position (2 non-redundant coords)
        + NODES_PER_EDGE * CUBE_EDGES   # Cube edge node position (1 non-redundant coord)
    ))

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
    for angle in range(0, 360*10):
        plot3d.view_init(30, angle)
        plt.draw()
        plt.pause(.002)

plot_metamaterial("local_test/image.png", *random_metamaterial())