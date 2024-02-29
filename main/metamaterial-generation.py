import numpy as np
import matplotlib.pyplot as plt

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
        The first entry in this tuple is a 2d numpy array where every row pair
        of values is the normalized 2D position of its corresponding node on its
        corresponding unit cube face. The nodes appear in the following order.
            Node 1: A coordinate (x,y) on the z=0 plane.
            Node 2: A coordinate (x,y) on the z=1 plane.
            Node 3: A coordinate (x,z) on the y=0 plane.
            Node 4: A coordinate (x,z) on the y=1 plane.
            Node 5: A coordinate (y,z) on the x=0 plane.
            Node 6: A coordinate (y,z) on the x=1 plane.
        
        The second entry in this tuple is a 2d numpy array edge adjacency
        matrix where a 1 at the i-th row and j-th column means that nodes
        i and j are connected by an edge. All other entries are 0.
        
        The third entry in this tuple is a 3d numpy array face adjacency
        tensor where a 1 at the (i,j,k) index means that nodes i, j, and k
        are connected by a triangular face. All other entries are 0.
    """
    
    # Generates the normalized [0,1) cube face node 2D positions
    node_pos = np.random.rand(NODES_PER_FACE * CUBE_FACES, 2)

    # Generates the edge adjacency matrix
    edge_adj_mat = np.random.rand(NUM_NODES, NUM_NODES)
    edge_adj_mat = ((edge_adj_mat + edge_adj_mat.T) < 2*0.5).astype(float) # Ensures symmetry

    # Prepares the base tensor for generating the face adjacency tensor
    og_face_adj_tensor = np.random.rand(NUM_NODES, NUM_NODES, NUM_NODES)

    # Accounts for each permutation of axes to make the adjacency tensor symmetric along all axes
    # KNOWN BUG: FACES DO NOT NECESSARILY AGREE WITH EDGE PRESENCE
    face_adj_tensor = np.copy(og_face_adj_tensor)
    face_adj_tensor += np.transpose(og_face_adj_tensor, (0,2,1))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (1,0,2))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (1,2,0))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (2,0,1))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (2,1,0))
    face_adj_tensor = (face_adj_tensor < 6*0.5).astype(float)

    return node_pos, edge_adj_mat, face_adj_tensor


def plot_metamaterial(filename, node_pos, edge_adj_mat, face_adj_tensor):
    """
    Plots the metamaterial with the given representation at the given filename.

    filename: str
        The name of the file at which the plot image will be saved.

    node_pos: ndarray
        The node position array as described in the specification of the
        random_metamaterial() function.

    edge_adj_mat: ndarray
        The edge adjacency matrix array as described in the specification
        of the random_metamaterial() function.

    face_adj_tensor: ndarray
        The face adjacency tensor array as described in the specification
        of the random_metamaterial() function.
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
            if not edge_adj_mat[n1, n2]:
                continue

            # Computes each n1 coordinate
            if n1 != 6:
                x1 = node_pos[n1, 0] if n1//2 != 2 else n1%2
                y1 = node_pos[n1, 1] if n1//2 == 0 else n1%2 if n1//2 == 1 else node_pos[n1, 0]
                z1 = node_pos[n1, 1] if n1//2 != 0 else n1%2
            else:
                x1 = y1 = z1 = 0.5

            # Computes each n2 coordinate
            if n2 != 6:
                x2 = node_pos[n2, 0] if n2//2 != 2 else n2%2
                y2 = node_pos[n2, 1] if n2//2 == 0 else n2%2 if n2//2 == 1 else node_pos[n2, 0]
                z2 = node_pos[n2, 1] if n2//2 != 0 else n2%2
            else:
                x2 = y2 = z2 = 0.5

            plot3d.plot([x1, x2], [y1, y2], zs=[z1, z2], linewidth=5)

            
    plt.savefig(filename)
