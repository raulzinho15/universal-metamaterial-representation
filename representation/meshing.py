import math
from representation.rep_class import *
from representation.generation import *

THICKNESS = 0.02
VERTICES_PER_EDGE = 8
VERTICES_PER_FACE = 3

def generate_edge_segment_mesh(point1, point2, next_point=None, prev_normal=None):
    """
    Generates the vertices and faces for an edge segment.
    
    point1: ndarray
        The position of the start of the edge.
    
    point2: ndarray
        The position of the end of the edge.

    Returns: (list of tuples of floats, list of tuples of ints)
        The first entry is a list of the vertex (x,y,z) coordinates.
        The second is a list containing each face's corresponding vertices,
        where the vertex numbers correspond to the index of the vertex in
        the first entry.
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

        # # Computes a random vector to be used for orthogonal vector generation for face 2
        # prev_normal = np.array([1,0,0])
        # if np.linalg.norm(np.cross(prev_normal, edge_dir)) < 0.1:
        #     prev_normal = np.array([0,1,0])
            
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
        [tuple(vertex) for vertex in face1_vertices] + [tuple(vertex) for vertex in face2_vertices], # Vertex coordinates
        [(0,1,2,3), (4,5,6,7)] + [(i, (i+1)%4, (i+1)%4+4, i+4) for i in range(4)] # Face vertex indices
    )



def generate_edge_mesh(material: Metamaterial, node1, node2):
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
    faces = []

    # Runs through each edge segment
    for edge in range(EDGE_SEGMENTS):
        vertex_list, face_list = generate_edge_segment_mesh(edge_points(edge), edge_points(edge+1), next_point=(None if edge+2 > EDGE_SEGMENTS else edge_points(edge+2)))

        # Adds the new vertices/faces
        vertices.extend(vertex_list)

        # Adds the faces from the edge
        for face in face_list:
            faces.append(tuple(map(lambda x: x+edge*VERTICES_PER_EDGE, face)))

    return vertices, faces


def generate_face_mesh(material: Metamaterial, node1, node2, node3):
    """
    Generates the vertices and faces for a face of the metamaterial.
    
    material: Metamaterial
        The metamaterial whose face will be meshed.

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
    
    face_points = material.compute_face_points(node1, node2, node3)

    vertices = []
    faces = []

    di = 0
    for t in range(EDGE_SEGMENTS):
        for s in range(EDGE_SEGMENTS-t):
            vertices.extend([tuple(face_points(s,t)), tuple(face_points(s+1,t)), tuple(face_points(s,t+1))])
            faces.append((di, di+1, di+2))
            di += 3
            if s+t+2 <= EDGE_SEGMENTS:
                vertices.append(tuple(face_points(s+1,t+1)))
                faces.append((di-2, di-1, di))
                di += 1

    return vertices, faces


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
    vertex_count = 0

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
                faces.append(tuple(map(lambda x: x + vertex_count, face)))

            vertex_count += len(vertex_list)

    # Generates the face mesh for each face
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
                    faces.append(tuple(map(lambda x: x + vertex_count, face)))

                vertex_count += len(vertex_list)

    return vertices, faces


def generate_metamaterial_grid_mesh(metamaterial: Metamaterial, shape=(1,1,1)):
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


def generate_metamaterials_zigzag_meshes(metamaterials: list[Metamaterial], shape=(1,1,1)):
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
    vertices, faces = [], []
    vertex_count = 0

    # Places each metamaterial along a zigzag
    for i, material in enumerate(metamaterials):

        # Stores the displacements
        dz = i//square_side
        dx = i%square_side if dz % 2 == 0 else (square_side-i-1)%square_side

        # Meshes the material
        material = material.translate(dx=(dx-square_side//2)*1.5*max(shape), dz=(dz-square_side//2)*1.5*max(shape))
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

