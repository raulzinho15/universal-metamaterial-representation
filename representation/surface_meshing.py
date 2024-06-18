import math
from representation.rep_class import *
from representation.generation import *

THICKNESS = 0.02
VERTICES_PER_EDGE = EDGE_SEGMENTS*2
VERTICES_PER_FACE = 6


def generate_edge_surface_mesh(material: Metamaterial, node1: int, node2: int) -> tuple[list[tuple], list[tuple]]:
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
        vertices.extend(
            tuple(edge_points[edge]
             + np.cos(2*np.pi*theta/EDGE_SEGMENTS) * normal1*THICKNESS
             + np.sin(2*np.pi*theta/EDGE_SEGMENTS) * normal2*THICKNESS)
                for theta in range(EDGE_SEGMENTS)
        )

        # # Adds the face at the ends of the cylindrical edge
        # if edge == 0 or edge == EDGE_SEGMENTS:
        #     faces.append(tuple(i+vertex_count for i in range(EDGE_SEGMENTS)))

        # Adds the faces that connect edge segments
        if edge != 0:
            faces.extend(
                (vertex_count-EDGE_SEGMENTS+i, vertex_count-EDGE_SEGMENTS+(i+1)%EDGE_SEGMENTS, vertex_count+(i+1)%EDGE_SEGMENTS, vertex_count+i)
                    for i in range(EDGE_SEGMENTS)
            )

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

    # Stores the axes for the hemisphere ends
    x_axes = [np.array(vertices[0])-edge_points[0], np.array(vertices[-EDGE_SEGMENTS])-edge_points[EDGE_SEGMENTS]]
    y_axes = [np.array(vertices[EDGE_SEGMENTS//4])-edge_points[0], np.array(vertices[EDGE_SEGMENTS//4-EDGE_SEGMENTS])-edge_points[EDGE_SEGMENTS]]
    z_axes = [edge_points[0]-edge_points[1], edge_points[-1]-edge_points[-2]]
    z_axes[0] *= THICKNESS/np.linalg.norm(z_axes[0])
    z_axes[1] *= THICKNESS/np.linalg.norm(z_axes[1])

    # Stores other hemisphere-specific data
    base_vertices_start = [0,len(vertices)-EDGE_SEGMENTS]
    centers = [edge_points[0], edge_points[-1]]

    # Runs through each hemisphere
    for h in range(2):

        # Runs through each theta
        for th in range(EDGE_SEGMENTS-1, -1, -1):
            
            # Handles the north pole
            if th == 0:

                # Computes the vertex
                vertices.append(tuple(centers[h]+z_axes[h]))

                # Adds the faces
                for i in range(EDGE_SEGMENTS):
                    ring_index = len(vertices)-EDGE_SEGMENTS-1
                    faces.append((ring_index+i, ring_index+(i+1)%EDGE_SEGMENTS, len(vertices)-1))

            # Handles a non-north pole circumference
            else:

                # Computes the vertices
                for phi in range(EDGE_SEGMENTS):

                    # Adjusts theta/phi
                    theta = th/EDGE_SEGMENTS * np.pi/2
                    phi = phi/EDGE_SEGMENTS * 2*np.pi

                    # Computes the vertex coordinates
                    x_coord = x_axes[h]*np.sin(theta)*np.cos(phi)
                    y_coord = y_axes[h]*np.sin(theta)*np.sin(phi)
                    z_coord = z_axes[h]*np.cos(theta)

                    # Stores the vertex coordinates
                    vertices.append(tuple(centers[h]+x_coord+y_coord+z_coord))

                # Adds the faces
                for i in range(EDGE_SEGMENTS):
                    ring1_index = len(vertices)-2*EDGE_SEGMENTS if th != EDGE_SEGMENTS-1 else base_vertices_start[h]
                    ring2_index = len(vertices)-EDGE_SEGMENTS
                    faces.append((ring1_index+i, ring1_index+(i+1)%EDGE_SEGMENTS, ring2_index+(i+1)%EDGE_SEGMENTS, ring2_index+i))

    return vertices, faces


def generate_face_surface_mesh(material: Metamaterial, node1: int, node2: int, node3: int) -> tuple[list[tuple], list[tuple]]:
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
    face_points_function = material.compute_face_points(node1, node2, node3)

    # Pre-computes the face points for faster computation
    face_points = [
        face_points_function(s,t)
            for s,t,u in BEZIER_TRIANGLE_PARAMETERS
    ]

    # Pre-computes the face normals for faster computation
    face_normals = []
    for s,t,u in BEZIER_TRIANGLE_PARAMETERS:

        # Stores the base point for the normal
        base_point = face_points[bezier_triangle_index(s,t)]

        # Stores the points from which the normal is deduced
        if s+t != EDGE_SEGMENTS:
            point1 = face_points[bezier_triangle_index(s+1,t)]
            point2 = face_points[bezier_triangle_index(s,t+1)]

        elif s == EDGE_SEGMENTS:
            point1 = face_points[bezier_triangle_index(s-1,t+1)]
            point2 = face_points[bezier_triangle_index(s-1,t)]

        elif t == EDGE_SEGMENTS:
            point1 = face_points[bezier_triangle_index(s,t-1)]
            point2 = face_points[bezier_triangle_index(s+1,t-1)]

        else:
            point1 = face_points[bezier_triangle_index(s-1,t+1)]
            point2 = face_points[bezier_triangle_index(s-1,t)]

        # Computes the face's normal
        face_normal = np.cross(point1-base_point, point2-base_point)
        face_normal /= np.linalg.norm(face_normal)

        # Stores the face's normal
        face_normals.append(face_normal*THICKNESS)

    # Stores the vertices and faces
    vertices = []
    faces = []

    # Stores each vertex
    for point, normal in zip(face_points, face_normals):
        vertices.append(tuple(point - normal)) # Bottom vertices
        vertices.append(tuple(point + normal)) # Top vertices

    # Runs through each top/bottom face
    for s in range(EDGE_SEGMENTS):
        for t in range(EDGE_SEGMENTS-s):

            # Stores the point indices
            point1_index = bezier_triangle_index(s,t)*2
            point2_index = bezier_triangle_index(s+1,t)*2
            point3_index = bezier_triangle_index(s,t+1)*2

            # Adds the faces
            faces.append((point1_index, point2_index, point3_index)) # Bottom face
            faces.append((point1_index+1, point2_index+1, point3_index+1)) # Top face

            # Checks for a second face
            if s+t+2 <= EDGE_SEGMENTS:

                # Stores the new point index
                point4_index = bezier_triangle_index(s+1,t+1)*2

                # Adds the faces
                faces.append((point2_index, point3_index, point4_index)) # Bottom face
                faces.append((point2_index+1, point3_index+1, point4_index+1)) # Top face

    # Runs through the edge faces
    for t in range(EDGE_SEGMENTS):
        s = EDGE_SEGMENTS-t # Computes the opposite parameter

        # Stores the u=0 edge faces
        faces.append((
            bezier_triangle_index(s,t)*2,
            bezier_triangle_index(s-1,t+1)*2,
            bezier_triangle_index(s-1,t+1)*2+1,
            bezier_triangle_index(s,t)*2+1,
        ))

        # Stores the t=0 edge faces
        faces.append((
            bezier_triangle_index(t,0)*2,
            bezier_triangle_index(t+1,0)*2,
            bezier_triangle_index(t+1,0)*2+1,
            bezier_triangle_index(t,0)*2+1,
        ))

        # Stores the s=0 edge faces
        faces.append((
            bezier_triangle_index(0,t)*2,
            bezier_triangle_index(0,t+1)*2,
            bezier_triangle_index(0,t+1)*2+1,
            bezier_triangle_index(0,t)*2+1,
        ))

    return vertices, faces


def generate_metamaterial_surface_mesh(material: Metamaterial) -> tuple[list[list[tuple]], list[list[tuple]]]:
    """
    Generates the mesh for the metamaterial.

    material: Metamaterial
        The material for which the mesh will be generated.

    Returns: `tuple[list[list[tuple]], list[list[tuple]]]`
        The first item contains the vertices of each component
        in the metamaterial mesh, where the first axis
        separates distinct components, and the second axis
        separates the coordinates of a component's vertices.

        The second item contains the faces of each component
        in the metamaterial mesh, where the first axis
        separates distinct components, and the second axis
        separates the indices of a component's faces.
    """

    # Stores the vertices and faces
    vertices, faces = [], []

    # Runs through each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips nodes without an edge between them
            if not material.has_edge(n1, n2):
                continue

            # Computes the edge's components
            edge_vertices, edge_faces = generate_edge_surface_mesh(material, n1, n2)

            # Stores the edge's components
            vertices.append(edge_vertices)
            faces.append(edge_faces)

    # Runs through each face
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):

                # Skips nodes without a face between them
                if not material.has_face(n1, n2, n3):
                    continue

                # Computes the face's components
                face_vertices, face_faces = generate_face_surface_mesh(material, n1, n2, n3)

                # Stores the face's components
                vertices.append(face_vertices)
                faces.append(face_faces)

    return vertices, faces


def generate_metamaterial_grid_surface_mesh(metamaterial: Metamaterial, shape=(1,1,1)) -> tuple[list[list[tuple]], list[list[tuple]]]:
    """
    Generates the mesh for the metamaterial.

    metamaterial: Metamaterial
        The material for which the mesh will be generated.

    shape: tuple of ints
        The amount in the x, y, and z directions to grid the metamaterial.
        If no gridding (i.e., just the metamaterial by itself), the input
        should be (1,1,1).

    Returns: `tuple[list[list[tuple]], list[list[tuple]]]`
        The first item contains the vertices of each component
        in the metamaterial mesh, where the first axis
        separates distinct components, and the second axis
        separates the coordinates of a component's vertices.

        The second item contains the faces of each component
        in the metamaterial mesh, where the first axis
        separates distinct components, and the second axis
        separates the indices of a component's faces.
    """

    # Computes the materials to mesh
    materials = metamaterial_grid(metamaterial, shape)

    # Stores values for the meshing
    vertices, faces = [], []

    # Runs through each material
    for material in materials:

        # Computes the material's components
        mat_vertices, mat_faces = generate_metamaterial_surface_mesh(material)

        # Stores the material's components
        vertices.extend(mat_vertices)
        faces.extend(mat_faces)

    return vertices, faces


def generate_metamaterials_zigzag_surface_meshes(metamaterials: list[Metamaterial], shape=(1,1,1)) -> tuple[list[tuple], list[tuple]]:
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


def optimize_vertices(vertices: list[tuple], faces: list[tuple]) -> tuple[list[tuple], list[tuple]]:

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


def save_obj(vertices: list[tuple], faces: list[tuple], filepath: str):
    """
    Saves the .obj file with the given vertices and face vertex indices.

    vertices: list of tuples
        The vertex (x,y,z) coordinates.

    faces: list of tuples
        The vertex indices corresponding to each face.

    filepath: str
        The path at which the file will be saved.
    """

    print("Saving:", filepath[filepath.rindex("/")+1:])

    with open(filepath, 'w') as f:

        # Writes each vertex
        for vertex in vertices:
            f.write(f"v {' '.join(map(str, map(lambda v: np.round(v,5), vertex)))}\n") # Rounds to 5 decimal places

        # Writes each face
        for face in faces:
            f.write("f")
            for vertex_index in face:
                f.write(f" {vertex_index + 1}") # 1-indexing
            f.write("\n")


def save_multi_obj(vertices: list[list[tuple]], faces: list[list[tuple]], filepath: str, precision=6, verbose=True):
    """
    Saves an .obj file with multiple objects.

    vertices: `list[list[tuple]]`
        The vertices of the different objects. The first
        axis separates the different objects. The second
        axis separates different vertices in the same
        object.

    faces: `list[list[tuple]]`
        The faces of the different objects. The first
        axis separates the different objects. The second
        axis separates different faces in the same
        object. Must have `len(faces) == len(vertices)`.
        The indices should be local to their own object.
        The indices should be 0-indexed.

    filepath: `str`
        The path at which the file will be saved.

    precision: `int`, optional
        The number of decimal places with which to save the
        vertex coordinates. Default is `6`.

    verbose: `bool`, optional
        Whether status messages will be printed to the console.
        Default is `True`.
    """

    # Tells the user the saving is beginning
    if verbose:
        print("Saving:", filepath[filepath.rindex("/")+1:])

    # Stores data for the file writing
    num_vertices = 0

    # Writes to the file
    with open(filepath, "w") as f:

        # Runs through each object
        for i, obj_parts in enumerate(zip(vertices, faces)):

            # Stores the parts of the object
            obj_vertices, obj_faces = obj_parts

            # Writes the object
            f.write(f"o Component{i}\n")

            # Writes each vertex
            for vertex in obj_vertices:
                f.write("v")
                for coord in vertex:
                    f.write(f" {np.round(coord, precision)}")
                f.write("\n")

            # Writes each face
            for face in obj_faces:
                f.write("f")
                for vertex_index in face:
                    f.write(f" {vertex_index + num_vertices + 1}")
                f.write("\n")

            # Updates the number of vertices
            num_vertices += len(obj_vertices)

    # Tells the user the save finihsed
    if verbose:
        print("Saved!")


