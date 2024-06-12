import math
from representation.rep_class import *
from representation.generation import *

THICKNESS = 0.02
VERTICES_PER_EDGE = 8
VERTICES_PER_FACE = 3

def generate_edge_segment_volume_mesh(point1, point2, next_point=None, prev_normal=None):
    """
    Generates the vertices and tetrahedra for an edge segment.
    
    point1: ndarray
        The position of the start of the edge segment.
    
    point2: ndarray
        The position of the end of the edge segment.

    Returns: (list of tuples of floats, list of tuples of ints)
        The first entry is a list of the vertex (x,y,z) coordinates.
        The second is a list containing each tetrahedron's corresponding
        vertices, where the vertex numbers correspond to the index of the
        vertex in the first entry.
    """

    # Computes the edge direction for face 1
    edge_dir = point1 - point2
    edge_len = np.linalg.norm(edge_dir)
    edge_dir /= edge_len

    # Computes a random vector to be used for orthogonal vector generation for face 1
    if prev_normal is None:
        prev_normal = np.array([1,0,0])
        if np.linalg.norm(np.cross(prev_normal, edge_dir)) < 0.1:
            prev_normal = np.array([0,1,0])
        
    # Computes two co-orthogonal vectors orthogonal to the edge direction for face1
    basis1 = np.cross(edge_dir, prev_normal)
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(edge_dir, basis1)
    basis2 /= np.linalg.norm(basis2)

    # Computes the face 1 (around node1) vertices
    face1_vertices = [
        point1 + basis1*THICKNESS + basis2*THICKNESS,
        point1 + basis1*THICKNESS - basis2*THICKNESS,
        point1 - basis1*THICKNESS - basis2*THICKNESS,
        point1 - basis1*THICKNESS + basis2*THICKNESS,
    ]

    # Adapts the normal to how it will appear on the next segment, if applicable
    if next_point is not None:

        # Computes the edge direction for face 2
        edge_dir = point2 - next_point
        edge_len = np.linalg.norm(edge_dir)
        edge_dir /= edge_len

        # Computes two co-orthogonal vectors orthogonal to the edge direction for face 2
        basis1 = np.cross(edge_dir, prev_normal)
        basis1 /= np.linalg.norm(basis1)
        basis2 = np.cross(edge_dir, basis1)
        basis2 /= np.linalg.norm(basis2)

    # Computes the face 2 (around node2) vertices
    face2_vertices = [
        point2 + basis1*THICKNESS + basis2*THICKNESS,
        point2 + basis1*THICKNESS - basis2*THICKNESS,
        point2 - basis1*THICKNESS - basis2*THICKNESS,
        point2 - basis1*THICKNESS + basis2*THICKNESS,
    ]

    return (
        basis2,
        [tuple(vertex) for vertex in face1_vertices] + [tuple(vertex) for vertex in face2_vertices], # Vertex coordinates
        [(0,1,4,3), (1,3,2,6), (1,4,5,6), (4,3,7,6)] # Tetrahedron vertex indices
    )


def generate_edge_volume_mesh(material: Metamaterial, node1, node2):
    """
    Generates the vertices and faces for an edge of the metamaterial.
    
    material: Metamaterial
        The metamaterial whose edge will be meshed.

    node1: int
        The node ID for one of the nodes defining the edge.

    node2: int
        The node ID for the other node defining the edge.

    Returns: (list of tuples of floats, list of tuples of ints)
        The first entry is a list of the vertex (x,y,z) coordinates.
        The second is a list containing each face's corresponding vertices,
        where the vertex numbers correspond to the index of the vertex in
        the first entry.
    """

    # Computes the points along the edge
    edge_points = material.compute_edge_points(node1, node2)

    # Stores values for the function
    vertices = []
    tetrahedra = []

    # Runs through each edge segment
    prev_normal = None
    for edge in range(EDGE_SEGMENTS):
        prev_normal, vertex_list, tetrahedra_list = generate_edge_segment_volume_mesh(edge_points(edge), edge_points(edge+1), next_point=(None if edge+2 > EDGE_SEGMENTS else edge_points(edge+2)), prev_normal=prev_normal)

        # Adds the new vertices/faces
        vertices.extend(vertex_list)

        # Adds the tetrahedra from the edge
        for tetrahedron in tetrahedra_list:
            tetrahedra.append(tuple(map(lambda x: x+edge*VERTICES_PER_EDGE, tetrahedron)))

    return vertices, tetrahedra


# def generate_face_mesh(material: Metamaterial, node1, node2, node3):
#     """
#     Generates the vertices and faces for a face of the metamaterial.
    
#     material: Metamaterial
#         The metamaterial whose face will be meshed.

#     node1: int
#         The node ID for one of the nodes defining the face.

#     node2: int
#         The node ID for one of the other nodes defining the face.

#     node3: int
#         The node ID of the last node defining the face.

#     Returns: (list of tuples of floats, list of tuples of ints)
#         The first entry is a list of the vertex (x,y,z) coordinates.
#         The second is a list containing each face's corresponding vertices,
#         where the vertex numbers correspond to the index of the vertex in
#         the first entry.
#     """
    
#     face_points = material.compute_face_points(node1, node2, node3)

#     vertices = []
#     faces = []

#     di = 0
#     for t in range(EDGE_SEGMENTS):
#         for s in range(EDGE_SEGMENTS-t):
#             vertices.extend([tuple(face_points(s,t)), tuple(face_points(s+1,t)), tuple(face_points(s,t+1))])
#             faces.append((di, di+1, di+2))
#             di += 3
#             if s+t+2 <= EDGE_SEGMENTS:
#                 vertices.append(tuple(face_points(s+1,t+1)))
#                 faces.append((di-2, di-1, di))
#                 di += 1

#     return vertices, faces


def generate_metamaterial_volume_mesh(material: Metamaterial):
    """
    Generates the mesh for the metamaterial.

    material: Metamaterial
        The material for which the mesh will be generated.

    Returns: (list of tuples, list of tuples)
        The vertex and face list, respectively, as described in 
        the save_obj() function's specification.
    """

    # Stores the vertices and faces
    vertices = []
    tetrahedra = []
    vertex_count = 0

    # Generates the edge mesh for each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips non-edges
            if not material.has_edge(n1, n2):
                continue

            # Adds the vertices from the edge
            vertex_list, tetrahedra_list = generate_edge_volume_mesh(material, n1, n2)
            vertices.extend(vertex_list)

            # Adds the faces from the edge
            for tetrahedron in tetrahedra_list:
                tetrahedra.append(tuple(map(lambda x: x + vertex_count, tetrahedron)))

            vertex_count += len(vertex_list)

    # # Generates the face mesh for each face
    # for n1 in range(NUM_NODES):
    #     for n2 in range(n1+1, NUM_NODES):
    #         for n3 in range(n2+1, NUM_NODES):

    #             # Skips non-faces
    #             if not material.has_face(n1, n2, n3):
    #                 continue

    #             # Adds the vertices from the face
    #             vertex_list, tetrahedra_list = generate_face_mesh(material, n1, n2, n3)
    #             vertices.extend(vertex_list)

    #             # Adds the faces from the face
    #             for tetrahedron in tetrahedra_list:
    #                 tetrahedra.append(tuple(map(lambda x: x + vertex_count, tetrahedron)))

    #             vertex_count += len(vertex_list)

    return vertices, tetrahedra


def generate_metamaterial_grid_volume_mesh(metamaterial: Metamaterial, shape=(1,1,1)):
    """
    Generates the mesh for the metamaterial.

    metamaterial: Metamaterial
        The material for which the mesh will be generated.

    shape: tuple of ints
        The amount in the x, y, and z directions to grid the metamaterial.
        If no gridding (i.e., just the metamaterial by itself), the input
        should be (1,1,1).

    Returns: (list of tuples, list of tuples)
        The vertex and face list, respectively, as described in 
        the save_obj() function's specification.
    """

    # Computes the materials to mesh
    materials = metamaterial_grid(metamaterial, shape)

    # Stores values for the meshing
    vertex_count = 0
    vertices, tetrahedra = [], []

    # Meshes each material
    for material in materials:
        next_vertices, next_tetrahedra = generate_metamaterial_volume_mesh(material)
        vertices.extend(next_vertices)
        tetrahedra.extend([tuple(map(lambda x: x+vertex_count, tetrahedron)) for tetrahedron in next_tetrahedra])
        vertex_count += len(next_vertices)

    return vertices, tetrahedra


def generate_metamaterials_zigzag_volume_meshes(metamaterials: list[Metamaterial], shape=(1,1,1)):
    """
    Generates meshes for the given metamaterials with the given shape in a
    zigzag pattern.

    metamaterials: list[Metamaterial]
        The metamaterials of which meshes will be generated.

    shape: tuple of ints
        The gridding to be done on the metamaterials. If (1,1,1), then only
        one of each metamaterial will be meshed.
    """
    
    # Finds the best square
    square_side = math.ceil(len(metamaterials)**0.5)

    # Stores values for the function
    vertices, tetrahedra = [], []
    vertex_count = 0

    # Places each metamaterial along a zigzag
    for i, material in enumerate(metamaterials):

        # Stores the displacements
        dz = i//square_side
        dx = i%square_side if dz % 2 == 0 else (square_side-i-1)%square_side

        # Meshes the material
        material = material.translate(dx=(dx-square_side//2)*1.5*max(shape), dz=(dz-square_side//2)*1.5*max(shape))
        next_vertices, next_tetrahedra = generate_metamaterial_grid_volume_mesh(material, shape)
        vertices.extend(next_vertices)
        tetrahedra.extend([tuple(map(lambda x: x+vertex_count, tetrahedron)) for tetrahedron in next_tetrahedra])
        vertex_count += len(next_vertices)

    return vertices, tetrahedra


def optimize_nodes(nodes, tetrahedra):

    # Stores data structures for removing duplicate nodes
    unique_nodes = 0
    new_nodes = []
    node_to_index = {}
    index_to_index = {}

    # Removes duplicate nodes
    for i,node in enumerate(nodes):

        # First time seeing node
        if node not in node_to_index:
            node_to_index[node] = unique_nodes
            unique_nodes += 1
            new_nodes.append(node)
            
        index_to_index[i] = node_to_index[node]

    # Stores the tetrahedra with new indices
    new_tetrahedra = [map(lambda x:index_to_index[x], tetrahedron) for tetrahedron in tetrahedra]

    return new_nodes, new_tetrahedra


def save_msh(nodes, tetrahedra, filepath):
    """
    Saves the .msh file with the given vertices and tetrahedron vertex
    indices.

    nodes: list of tuples
        The node (x,y,z) coordinates.

    tetrahedra: list of tuples
        The vertex indices corresponding to each tetrahedron.

    filepath: str
        The path at which the file will be saved.
    """

    # Optimizes the node count
    nodes, tetrahedra = optimize_nodes(nodes, tetrahedra)

    # Writes the new file
    with open(filepath, 'w') as f:

        # Writes the mesh format lines
        f.write("$MeshFormat\n")
        f.write("2.0 0 8\n")
        f.write("$EndMeshFormat\n")

        # Writes each node
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for i,node in enumerate(nodes):
            f.write(f"{i+1} {' '.join(map(str, map(lambda v: np.round(v,4), node)))}\n") # Rounds to 4 decimal places
        f.write("$EndNodes\n")

        # Writes each tetrahedron
        f.write("$Elements\n")
        f.write(f"{len(tetrahedra)}\n")
        for i,tetrahedron in enumerate(tetrahedra):
            f.write(f"{i+1} 4 0 {' '.join(map(str, map(lambda x:x+1, tetrahedron)))}\n")
        f.write("$EndElements\n")