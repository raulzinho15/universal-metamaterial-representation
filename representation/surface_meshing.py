import math
from representation.rep_class import *
from representation.generation import *

THICKNESS = 0.02
VERTICES_PER_EDGE = EDGE_SEGMENTS*2
VERTICES_PER_FACE = 6


def generate_edge_surface_mesh(material: Metamaterial, node1, node2):
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
    edge_points_function = material.compute_edge_points(node1, node2)

    # Stores the edge points for faster computation
    edge_points = [edge_points_function(edge) for edge in range(EDGE_SEGMENTS+1)]

    # Stores vertex and face values for the function
    vertices = []
    faces = []

    # Computes a starting point for the normal vectors
    init_dir = edge_points[1] - edge_points[0]
    init_guess = np.array([1,0,0])
    if np.linalg.norm(np.cross(init_guess, init_dir)) < 1e-4:
        init_guess = np.array([0,1,0])
        
    # Computes the initial normals
    normal1 = np.cross(init_dir, init_guess)
    normal1 /= np.linalg.norm(normal1)
    normal2 = np.cross(init_dir, normal1)
    normal2 /= np.linalg.norm(normal2)

    # Stores the number of vertices seen so far
    vertex_count = 0

    # Runs through each edge point
    for edge in range(EDGE_SEGMENTS+1):

        # Adds all of the vertices on the circular face of this edge point
        vertices.extend([
            tuple(edge_points[edge]
             + np.cos(2*np.pi*theta/EDGE_SEGMENTS) * normal1*THICKNESS
             + np.sin(2*np.pi*theta/EDGE_SEGMENTS) * normal2*THICKNESS)
                for theta in range(EDGE_SEGMENTS)
        ])

        # Adds the face at the ends of the cylindrical edge
        if edge == 0 or edge == EDGE_SEGMENTS:
            faces.append(tuple(i+vertex_count for i in range(EDGE_SEGMENTS)))

        # Adds the faces that connect edge segments
        if edge != 0:
            faces.extend([
                (vertex_count-EDGE_SEGMENTS+i, vertex_count-EDGE_SEGMENTS+(i+1)%EDGE_SEGMENTS, vertex_count+(i+1)%EDGE_SEGMENTS, vertex_count+i)
                    for i in range(EDGE_SEGMENTS)
            ])

        # Updates the normals
        if edge != EDGE_SEGMENTS:

            # Computes the direction of the edge segment to be formed
            edge_dir = edge_points[edge+1] - edge_points[edge]

            # Computes the new normals
            normal2 = np.cross(edge_dir, normal1)
            normal2 /= np.linalg.norm(normal2)
            normal1 = np.cross(normal2, edge_dir)
            normal1 /= np.linalg.norm(normal1)

        # Updates the vertex count
        vertex_count += EDGE_SEGMENTS

    return vertices, faces


def generate_face_segment_surface_mesh(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray, face_normal: np.ndarray):
    """
    Generates the vertices and faces for an face segment.
    
    point1: ndarray
        One of the three nodes of the face.
    
    point2: ndarray
        Another of the three nodes of the face.
    
    point3: ndarray
        The last of the three nodes of the face.

    face_normal: np.ndarray
        The normal to the face. One normal is used for the whole face.

    Returns: (list of tuples of floats, list of tuples of ints)
        The first entry is a list of the vertex (x,y,z) coordinates.
        The second is a list containing each face's corresponding vertices,
        where the vertex numbers correspond to the index of the vertex in
        the first entry.
    """

    # Computes all vertices of the face segment
    segment_vertices = [
        point1 - face_normal*THICKNESS,
        point2 - face_normal*THICKNESS,
        point3 - face_normal*THICKNESS,
        point1 + face_normal*THICKNESS,
        point2 + face_normal*THICKNESS,
        point3 + face_normal*THICKNESS,
    ]

    # Returns the vertices and faces
    return (
        [tuple(vertex) for vertex in segment_vertices],
        [(0,1,2), (3,4,5), (0,1,4,3), (1,2,5,4), (2,0,3,5)]
    )


def generate_face_surface_mesh(material: Metamaterial, node1, node2, node3):
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
    
    # Creates function for computing face points
    face_points = material.compute_face_points(node1, node2, node3)

    # Stores the vertices and faces
    vertices = []
    faces = []

    # Stores the number of vertices so far
    vertex_count = 0

    # Computes the face normal
    face_normal = np.cross(face_points(0,EDGE_SEGMENTS) - face_points(EDGE_SEGMENTS, 0), face_points(0,0) - face_points(EDGE_SEGMENTS, 0))
    face_normal /= np.linalg.norm(face_normal)

    # Runs through each face
    for t in range(EDGE_SEGMENTS):
        for s in range(EDGE_SEGMENTS-t):

            # Gets the first face's vertices and faces
            vertex_list, face_list = generate_face_segment_surface_mesh(face_points(s,t), face_points(s+1,t), face_points(s,t+1), face_normal)
            
            # Stores the vertices
            vertices.extend(vertex_list)

            # Stores the faces
            for face in face_list:
                faces.append(tuple(map(lambda x:x+vertex_count, face)))
            vertex_count += VERTICES_PER_FACE

            # Checks for a second face
            if s+t+2 <= EDGE_SEGMENTS:

                # Gets the second face's vertices and faces
                vertex_list, face_list = generate_face_segment_surface_mesh(face_points(s+1,t+1), face_points(s,t+1), face_points(s+1,t), face_normal)
                
                # Stores the vertices
                vertices.extend(vertex_list)

                # Stores the faces
                for face in face_list:
                    faces.append(tuple(map(lambda x:x+vertex_count, face)))
                vertex_count += VERTICES_PER_FACE
    
    return vertices, faces


def generate_metamaterial_surface_mesh(material: Metamaterial):
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
            vertex_list, face_list = generate_edge_surface_mesh(material, n1, n2)
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
                vertex_list, face_list = generate_face_surface_mesh(material, n1, n2, n3)
                vertices.extend(vertex_list)

                # Adds the faces from the face
                for face in face_list:
                    faces.append(tuple(map(lambda x: x + vertex_count, face)))

                vertex_count += len(vertex_list)

    return vertices, faces


def generate_metamaterial_grid_surface_mesh(metamaterial: Metamaterial, shape=(1,1,1)):
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
        next_vertices, next_faces = generate_metamaterial_surface_mesh(material)
        vertices.extend(next_vertices)
        faces.extend([tuple(map(lambda x: x+vertex_count, face)) for face in next_faces])
        vertex_count += len(next_vertices)

    return vertices, faces


def generate_metamaterials_zigzag_surface_meshes(metamaterials: list[Metamaterial], shape=(1,1,1)):
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
        next_vertices, next_faces = generate_metamaterial_grid_surface_mesh(material, shape)
        vertices.extend(next_vertices)
        faces.extend([tuple(map(lambda x: x+vertex_count, face)) for face in next_faces])
        vertex_count += len(next_vertices)

    return vertices, faces


def optimize_vertices(vertices, faces):

    # Stores data structures for removing duplicate vertices
    unique_vertices = 0
    new_vertices = []
    vertex_to_index = {}
    index_to_index = {}

    # Removes duplicate vertices
    for i,vertex in enumerate(vertices):

        # First time seeing node
        if vertex not in vertex_to_index:
            vertex_to_index[vertex] = unique_vertices
            unique_vertices += 1
            new_vertices.append(vertex)
            
        index_to_index[i] = vertex_to_index[vertex]

    # Stores the faces with new indices
    new_faces = [map(lambda x:index_to_index[x], face) for face in faces]

    return new_vertices, new_faces


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

    vertices, faces = optimize_vertices(vertices, faces)

    with open(filepath, 'w') as f:

        # Writes each vertex
        for vertex in vertices:
            f.write(f"v {' '.join(map(str, map(lambda v: np.round(v,4), vertex)))}\n") # Rounds to 4 decimal places

        # Writes each face
        for face in faces:
            f.write("f")
            for vertex_index in face:
                f.write(f" {vertex_index + 1}") # 1-indexing
            f.write("\n")

