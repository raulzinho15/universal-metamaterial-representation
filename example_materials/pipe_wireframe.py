from representation.rep_class import *
from representation.rep_utils import *

# Prepares the node positions of the metamaterial
node_pos = np.zeros(NODE_POS_SIZE)

# Computes the node positions of the metamaterial
node_pos[0:12] = np.concatenate([
    euclidian_to_spherical(0,1,-1),
    euclidian_to_spherical(-1,1,0),
    euclidian_to_spherical(-1,0,1),
    euclidian_to_spherical(0,-1,1),
    euclidian_to_spherical(1,-1,0),
    euclidian_to_spherical(1,0,-1),
])
cube_node_pos = np.array([project_onto_cube(*spherical_to_euclidian(node_pos[i*2]*np.pi, node_pos[i*2+1]*2*np.pi)) for i in range(NODE_POS_SIZE//2)])

# Prepares the edge adjacencies of the metamaterial
edge_adj = np.zeros(EDGE_ADJ_SIZE)

# Prepares the edge parameters of the metamaterial
edge_params = np.zeros(EDGE_PARAMS_SIZE)

# Computes the edge adjacencies/parameters of the metamaterial
for n1 in range(6):
    n2 = (n1+1)%6

    # Sets up the edge adjacency
    edge_index = edge_adj_index(n1, n2)
    edge_adj[edge_index] = 1
    edge_index *= EDGE_BEZIER_COORDS

    # Computes the circle center around which the edge goes
    if n1 % 2 == 0:
        circle_center = np.zeros((1,3))
        circle_center[0,(n1//2+1)%3] = 1
    else:
        circle_center = np.ones((1,3))
        circle_center[0,(n1//2)%3] = 0

    # Computes the cos() and sin() coefficients
    n1, n2 = sorted((n1, n2))
    cos_coeff = cube_node_pos[n1] - circle_center
    sin_coeff = cube_node_pos[n2] - circle_center

    # Defines the function producing edge points
    def edge_function(t):

        # Normalizes the Bezier parameter
        t /= EDGE_SEGMENTS
        t *= np.pi/2

        # Returns the interpolated edge point
        return circle_center + cos_coeff*np.cos(t) + sin_coeff*np.sin(t)
    
    # Computes the edge parameters
    fit_edge_params = find_edge_params(edge_function)

    # Stores the edge parameters
    edge_params[edge_index : edge_index+EDGE_BEZIER_COORDS] = fit_edge_params

# Prepares the face adjacencies of the metamaterial
face_adj = np.zeros(FACE_ADJ_SIZE)

# Prepares the face parameters of the metamaterial
face_params = np.zeros(FACE_PARAMS_SIZE)

# Creates the metamaterial
PIPE_WIREFRAME = Metamaterial(node_pos, edge_adj, edge_params, face_adj, face_params)