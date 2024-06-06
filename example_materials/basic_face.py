from representation.rep_class import *
from representation.rep_utils import *

# Prepares the node positions of the metamaterial
node_pos = np.zeros(NODE_POS_SIZE)
node_pos[0:2] = euclidian_to_spherical(-1,-1,-1)
node_pos[2:4] = euclidian_to_spherical(1,-1,-1)
node_pos[4:6] = euclidian_to_spherical(-1,-1,1)

# Prepares the edge adjacencies of the metamaterial
edge_adj = np.zeros(EDGE_ADJ_SIZE)

# Prepares the metamaterial's face function
def face_function(s,t):
    return (
        s/EDGE_SEGMENTS * np.array([[0., 0., 0.]]) +
        t/EDGE_SEGMENTS * np.array([[1., 0., 0.]]) +
        (EDGE_SEGMENTS-s-t)/EDGE_SEGMENTS * np.array([[0., 0., 1.]]) +
        s*t*(EDGE_SEGMENTS-s-t)/EDGE_SEGMENTS**2/2 * np.array([[0., 1., 0.]])
    )

# Separately computes the edge parameters of the metamaterial
edge1_index = edge_adj_index(0,1) * EDGE_BEZIER_COORDS
edge1_function = lambda t: np.array([[t/EDGE_SEGMENTS,0,0]])
auto_edge1_params = find_edge_params(edge1_function)
edge2_index = edge_adj_index(0,2) * EDGE_BEZIER_COORDS
edge2_function = lambda t: np.array([[0,0,t/EDGE_SEGMENTS]])
auto_edge2_params = find_edge_params(edge2_function)
edge3_index = edge_adj_index(1,2) * EDGE_BEZIER_COORDS
edge3_function = lambda t: np.array([[1-t/EDGE_SEGMENTS,0,t/EDGE_SEGMENTS]])
auto_edge3_params = find_edge_params(edge3_function)

# Separately computes the face parameters of the metamaterial
face_index = face_adj_index(0,1,2) * EDGE_BEZIER_COORDS
all_edge_params = np.concatenate([
    auto_edge1_params,
    auto_edge2_params,
    auto_edge3_params,
]).reshape((EDGE_BEZIER_COORDS, 3))
auto_face_params = find_face_params(all_edge_params, face_function)

# Simultaneously computes the edge and face parameters of the metamaterial (overrides separately computed parameters)
auto_face_params, auto_edge1_params, auto_edge2_params, auto_edge3_params = find_face_params(None, face_function)

# Prepares the edge parameters of the metamaterial
edge_params = np.zeros(EDGE_PARAMS_SIZE)
edge_params[edge1_index : edge1_index + EDGE_BEZIER_COORDS] = auto_edge1_params
edge_params[edge2_index : edge2_index + EDGE_BEZIER_COORDS] = auto_edge2_params
edge_params[edge3_index : edge3_index + EDGE_BEZIER_COORDS] = auto_edge3_params

# Prepares the face adjacencies of the metamaterial
face_adj = np.zeros(FACE_ADJ_SIZE)
face_adj[face_adj_index(0,1,2)] = 1

# Prepares the face parameters of the metamaterial
face_params = np.zeros(FACE_PARAMS_SIZE)
face_params[face_index : face_index + FACE_BEZIER_COORDS] = auto_face_params

# Creates the metamaterial
BASIC_FACE = Metamaterial(node_pos, edge_adj, edge_params, face_adj, face_params)