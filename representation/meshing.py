import math
from representation.rep_class import *
from representation.generation import *

THICKNESS = 0.02
VERTICES_PER_EDGE = 8
VERTICES_PER_FACE = 6

def generate_edge_mesh(material: Metamaterial, node1, node2):
    """
    Generates the vertices and faces for an edge of the metamaterial.
    
    material: Metamaterial
        The metamaterial with the edge

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

    # Computes the edge direction
    node_pos1 = material.get_node_position(node1)
    node_pos2 = material.get_node_position(node2)
    edge_dir = node_pos1 - node_pos2
    edge_len = np.linalg.norm(edge_dir)
    edge_dir /= edge_len

    # Computes a random vector to be used for orthogonal vector generation
    rand_vec = np.array([1,0,0])
    if np.linalg.norm(np.cross(rand_vec, edge_dir)) < 0.1:
        rand_vec = np.array([0,1,0])
        
    # Computes two co-orthogonal vectors orthogonal to the edge direction
    basis1 = np.cross(edge_dir, rand_vec)
    basis2 = np.cross(edge_dir, basis1)

    # Computes the face 1 (around node1) vertices
    face1_vertices = [
        node_pos1 + basis1*THICKNESS + basis2*THICKNESS,
        node_pos1 + basis1*THICKNESS - basis2*THICKNESS,
        node_pos1 - basis1*THICKNESS - basis2*THICKNESS,
        node_pos1 - basis1*THICKNESS + basis2*THICKNESS,
    ]

    # Computes the face 2 (around node2) vertices
    face2_vertices = [
        node_pos2 + basis1*THICKNESS + basis2*THICKNESS,
        node_pos2 + basis1*THICKNESS - basis2*THICKNESS,
        node_pos2 - basis1*THICKNESS - basis2*THICKNESS,
        node_pos2 - basis1*THICKNESS + basis2*THICKNESS,
    ]

    return (
        [tuple(vertex) for vertex in face1_vertices] + [tuple(vertex) for vertex in face2_vertices], # Vertex coordinates
        [(0,1,2,3), (4,5,6,7)] + [(i, (i+1)%4, (i+1)%4+4, i+4) for i in range(4)] # Face vertex indices
    )


def generate_face_mesh(material: Metamaterial, node1, node2, node3):
    """
    Generates the vertices and faces for a face of the metamaterial.
    
    material: Metamaterial
        The metamaterial with the edge

    node1: int
        The node ID for one of the nodes defining the face.

    node2: int
        The node ID for one of the other nodes defining the face.

    node3: int
        The node ID of the last node defining the face.

    Returns: (list of tuples of floats, list of tuples of ints)
        The first entry is a list of the vertex (x,y,z) coordinates.
        The second is a list containing each face's corresponding vertices,
        where the vertex numbers correspond to the index of the vertex in
        the first entry.
    """
    
    # Computes the nodes' positions
    node_pos1 = material.get_node_position(node1)
    node_pos2 = material.get_node_position(node2)
    node_pos3 = material.get_node_position(node3)

    # Computes the normal to the triangle
    tri_norm = np.cross(node_pos2-node_pos1, node_pos3-node_pos1)
    tri_norm /= np.linalg.norm(tri_norm)

    # Computes the top face vertices
    face1_vertices = [
        node_pos1 + tri_norm * THICKNESS,
        node_pos2 + tri_norm * THICKNESS,
        node_pos3 + tri_norm * THICKNESS,
    ]

    # Computes the bottom face vertices
    face2_vertices = [
        node_pos1 - tri_norm * THICKNESS,
        node_pos2 - tri_norm * THICKNESS,
        node_pos3 - tri_norm * THICKNESS,
    ]
    
    return (
        [tuple(vertex) for vertex in face1_vertices] + [tuple(vertex) for vertex in face2_vertices], # Vertex coordinates
        [(0,1,2), (3,4,5)] + [(i, (i+1)%3, (i+1)%3+3, i+3) for i in range(3)] # Face vertex indices
    )


def generate_metamaterial_mesh(material: Metamaterial):
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
    faces = []
    edge_count = 0

    # Generates the edge mesh for each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips non-edges
            if not material.has_edge(n1, n2):
                continue

            # Adds the vertices from the edge
            vertex_list, face_list = generate_edge_mesh(material, n1, n2)
            vertices.extend(vertex_list)

            # Adds the faces from the edge
            for face in face_list:
                faces.append(tuple(map(lambda x: x+edge_count*VERTICES_PER_EDGE, face)))

            edge_count += 1

    # Generates the face mesh for each face
    face_count = 0
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):

                # Skips non-faces
                if not material.has_face(n1, n2, n3):
                    continue

                # Adds the vertices from the face
                vertex_list, face_list = generate_face_mesh(material, n1, n2, n3)
                vertices.extend(vertex_list)

                # Adds the faces from the face
                for face in face_list:
                    faces.append(tuple(map(lambda x: x + edge_count*VERTICES_PER_EDGE + face_count*VERTICES_PER_FACE, face)))

                face_count += 1

    return vertices, faces


def generate_metamaterial_grid_mesh(metamaterial: Metamaterial, shape):
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
    vertices, faces = [], []

    # Meshes each material
    for material in materials:
        next_vertices, next_faces = generate_metamaterial_mesh(material)
        vertices.extend(next_vertices)
        faces.extend([tuple(map(lambda x: x+vertex_count, face)) for face in next_faces])
        vertex_count += len(next_vertices)

    return vertices, faces


def generate_metamaterials_zigzag_meshes(metamaterials: list[Metamaterial], shape):
    
    # Finds the best square
    square_side = math.ceil(len(metamaterials)**0.5)

    # Stores values for the function
    vertices, faces = [], []
    vertex_count = 0

    # Places each metamaterial along a zigzag
    for i, material in enumerate(metamaterials):

        # Stores the displacements
        dz = i//square_side
        dx = i%square_side if dz % 2 == 0 else (square_side-i-1)%square_side

        # Meshes the material
        material = material.translate(dx=dx*1.2*max(shape), dz=dz*1.2*max(shape))
        next_vertices, next_faces = generate_metamaterial_grid_mesh(material, shape)
        vertices.extend(next_vertices)
        faces.extend([tuple(map(lambda x: x+vertex_count, face)) for face in next_faces])
        vertex_count += len(next_vertices)

    return vertices, faces


def save_obj(vertices, faces, filepath):
    """
    Saves the .obj file with the given vertices and face vertex indices.

    vertices: list of tuples
        The vertex (x,y,z) coordinates.

    faces: list of tuples
        The vertex indices corresponding to each face.

    filepath: str
        The path at which the file will be saved.
    """
    with open(filepath, 'w') as f:

        # Writes each vertex
        for vertex in vertices:
            f.write(f"v {' '.join(map(str, vertex))}\n")

        # Writes each face
        for face in faces:
            f.write("f")
            for vertex_index in face:
                f.write(f" {vertex_index + 1}") # 1-indexing
            f.write("\n")

