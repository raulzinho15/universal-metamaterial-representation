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
        The first entry in this tuple is a 1d numpy array where every pair
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
    face_adj_tensor = np.copy(og_face_adj_tensor)
    face_adj_tensor += np.transpose(og_face_adj_tensor, (0,2,1))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (1,0,2))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (1,2,0))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (2,0,1))
    face_adj_tensor += np.transpose(og_face_adj_tensor, (2,1,0))
    face_adj_tensor = (face_adj_tensor < 6*0.5).astype(float)

    return node_pos, edge_adj_mat, face_adj_tensor
