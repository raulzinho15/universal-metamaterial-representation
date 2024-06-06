from representation.rep_class import *
from representation.rep_utils import *

# Prepares the node positions of the metamaterial
node_pos = np.zeros(NODE_POS_SIZE)

# Computes the node positions of the metamaterial
node_pos[0:2] =   euclidian_to_spherical(1,-1,0)
node_pos[2:4] =   euclidian_to_spherical(0,-1,1)
node_pos[4:6] =   euclidian_to_spherical(-1,0,1)
node_pos[6:8] =   euclidian_to_spherical(-1,1,0)
node_pos[8:10] =  euclidian_to_spherical(0,1,-1)
node_pos[10:12] = euclidian_to_spherical(1,0,-1)
cube_node_pos = np.array([project_onto_cube(*spherical_to_euclidian(node_pos[i*2]*np.pi, node_pos[i*2+1]*2*np.pi)) for i in range(NODE_POS_SIZE//2)])

# Prepares the edge adjacencies of the metamaterial
edge_adj = np.zeros(EDGE_ADJ_SIZE)

# Prepares the edge parameters of the metamaterial
edge_params = np.zeros(EDGE_PARAMS_SIZE)

# Prepares the face adjacencies of the metamaterial
face_adj = np.zeros(FACE_ADJ_SIZE)

# Prepares the face parameters of the metamaterial
face_params = np.zeros(FACE_PARAMS_SIZE)

# Computes the edge/face parameters of the metamaterial
n1 = 0
for n2 in range(1,5):
    n3 = n2+1

    # Sets up the face adjacency
    face_index = face_adj_index(n1, n2, n3)
    face_adj[face_index] = 1
    face_index *= FACE_BEZIER_COORDS

    # Computes the best-fit edge/face parameters
    fit_face_params, fit_edge1_params, fit_edge2_params, fit_edge3_params = flat_face_params(cube_node_pos[n1], cube_node_pos[n2], cube_node_pos[n3])

    # Stores the face parameters
    face_params[face_index : face_index + FACE_BEZIER_COORDS] = fit_face_params

    # Computes the edge parameter indices
    edge1_index = edge_adj_index(n1, n2) * EDGE_BEZIER_COORDS
    edge2_index = edge_adj_index(n1, n3) * EDGE_BEZIER_COORDS
    edge3_index = edge_adj_index(n2, n3) * EDGE_BEZIER_COORDS

    # Stores the edge parameters
    edge_params[edge1_index : edge1_index + EDGE_BEZIER_COORDS] = fit_edge1_params
    edge_params[edge2_index : edge2_index + EDGE_BEZIER_COORDS] = fit_edge2_params
    edge_params[edge3_index : edge3_index + EDGE_BEZIER_COORDS] = fit_edge3_params

# Creates the metamaterial
HEXAGON_SHELL = Metamaterial(node_pos, edge_adj, edge_params, face_adj, face_params)