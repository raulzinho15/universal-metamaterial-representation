from representation.rep_class import *
from representation.rep_utils import *

# Prepares the node positions of the metamaterial
node_pos = np.zeros(NODE_POS_SIZE)
node_pos[0:2] = euclidian_to_spherical(0,0,-1)
node_pos[2:4] = euclidian_to_spherical(0,0,1)

# Prepares the edge adjacencies of the metamaterial
edge_adj = np.zeros(EDGE_ADJ_SIZE)
index = edge_adj_index(0,1)
edge_adj[index] = 1

# Prepares the edge parameters of the metamaterial
edge_params = np.zeros(EDGE_PARAMS_SIZE)
index *= EDGE_BEZIER_COORDS
edge_function = lambda t: np.array([[0.5 + np.sin(t/EDGE_SEGMENTS*2*np.pi)/2, 0.5 + (t/EDGE_SEGMENTS)*(1-t/EDGE_SEGMENTS)*2, t/EDGE_SEGMENTS]])
edge_params[index : index + EDGE_BEZIER_COORDS] = find_edge_params(edge_function)

# Prepares the face adjacencies of the metamaterial
face_adj = np.zeros(FACE_ADJ_SIZE)

# Prepares the face parameters of the metamaterial
face_params = np.zeros(FACE_PARAMS_SIZE)

# Creates the metamaterial
BASIC_LINE = Metamaterial(node_pos, edge_adj, edge_params, face_adj, face_params)