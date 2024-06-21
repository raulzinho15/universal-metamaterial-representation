import math
import trimesh
from representation.rep_class import *
from representation.generation import *

THICKNESS = 0.02
VERTICES_PER_EDGE = EDGE_SEGMENTS*2
VERTICES_PER_FACE = 6


def generate_node_surface_mesh(material: Metamaterial, node: int) -> tuple[list[tuple], list[tuple]]:
    """
    Generates the vertices and faces for a node of the metamaterial.
    
    material: Metamaterial
        The metamaterial whose edge will be meshed.

    node1: int
        The node ID for the target node.
    """

    # Computes the node position
    center = material.get_node_position(node)

    # Stores the mesh values
    vertices, faces = [], []

    # Geometric properties of the node mesh
    node_segments = EDGE_SEGMENTS//2
    sphere_radius = THICKNESS*51/50

    # Computes angle values
    def thetas():
        for i in range(node_segments+1):
            yield i, i/node_segments * np.pi
    def phis():
        for i in range(node_segments):
            yield i, i/node_segments * 2*np.pi

    # Runs through each point on the surface of the sphere
    for i,theta in thetas():

        # Computes the z Euclidean coordinate
        z = np.array([0.,0.,1.]) * sphere_radius * np.cos(theta)

        # Handles all other points in between
        for j,phi in phis():

            # Computes the Euclidean coordinates
            x = np.array([1.,0.,0.]) * sphere_radius * np.sin(theta) * np.cos(phi)
            y = np.array([0.,1.,0.]) * sphere_radius * np.sin(theta) * np.sin(phi)

            # Stores the vertices along the ring
            vertices.append(tuple(center+x+y+z))

            # Skips redundantly computing the poles
            if i == 0 or i == node_segments:
                break

        # Computes the face vertices
        for j in range(node_segments):

            # Skips when only the north pole point has been added
            if i == 0:
                break

            # Handles the faces touching the north pole
            if i == 1:
                faces.append((0, 1 + j, 1 + (j+1)%node_segments))
                continue

            # Handles the faces touching the south pole
            if i == node_segments:
                offset = len(vertices)-node_segments-1
                faces.append((offset + (j+1)%node_segments, offset + j, offset + node_segments))
                continue

            # Handles all other faces
            ring1_index = len(vertices)-2*node_segments
            ring2_index = len(vertices)-node_segments
            face1 = (ring2_index + (j+1)%node_segments, ring1_index + (j+1)%node_segments, ring1_index + j)
            face2 = (ring1_index + j, ring2_index + j, ring2_index + (j+1)%node_segments)
            faces.append(face1)
            faces.append(face2)

    return vertices, faces


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

        # Adds the face at the ends of the cylindrical edge
        if edge == 0:
            faces.extend((vertex_count+i+1, vertex_count+i, vertex_count) for i in range(1,EDGE_SEGMENTS-1))
        elif edge == EDGE_SEGMENTS:
            faces.extend((vertex_count, vertex_count+i, vertex_count+i+1) for i in range(1,EDGE_SEGMENTS-1))

        # Adds the faces that connect edge segments
        if edge != 0:
            for i in range(EDGE_SEGMENTS):
                face1 = (vertex_count-EDGE_SEGMENTS+i, vertex_count-EDGE_SEGMENTS+(i+1)%EDGE_SEGMENTS, vertex_count+(i+1)%EDGE_SEGMENTS)
                face2 = (vertex_count+(i+1)%EDGE_SEGMENTS, vertex_count+i, vertex_count-EDGE_SEGMENTS+i)
                faces.append(face1)
                faces.append(face2)

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
            faces.append((point3_index, point2_index, point1_index)) # Bottom face
            faces.append((point1_index+1, point2_index+1, point3_index+1)) # Top face

            # Checks for a second face
            if s+t+2 <= EDGE_SEGMENTS:

                # Stores the new point index
                point4_index = bezier_triangle_index(s+1,t+1)*2

                # Adds the faces
                faces.append((point2_index, point3_index, point4_index)) # Bottom face
                faces.append((point4_index+1, point3_index+1, point2_index+1)) # Top face

    # Runs through the edge faces
    for t in range(EDGE_SEGMENTS):
        s = EDGE_SEGMENTS-t # Computes the opposite parameter

        # Stores the u=0 edge faces
        faces.append((
            bezier_triangle_index(s,t)*2,
            bezier_triangle_index(s-1,t+1)*2,
            bezier_triangle_index(s-1,t+1)*2+1,
        ))
        faces.append((
            bezier_triangle_index(s-1,t+1)*2+1,
            bezier_triangle_index(s,t)*2+1,
            bezier_triangle_index(s,t)*2,
        ))

        # Stores the t=0 edge faces
        faces.append((
            bezier_triangle_index(t,0)*2,
            bezier_triangle_index(t+1,0)*2,
            bezier_triangle_index(t+1,0)*2+1,
        ))
        faces.append((
            bezier_triangle_index(t+1,0)*2+1,
            bezier_triangle_index(t,0)*2+1,
            bezier_triangle_index(t,0)*2,
        ))

        # Stores the s=0 edge faces
        faces.append((
            bezier_triangle_index(0,t+1)*2+1,
            bezier_triangle_index(0,t+1)*2,
            bezier_triangle_index(0,t)*2,
        ))
        faces.append((
            bezier_triangle_index(0,t)*2,
            bezier_triangle_index(0,t)*2+1,
            bezier_triangle_index(0,t+1)*2+1,
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

    # Stores the nodes/edges that are used
    used_nodes = set()
    used_edges = set()

    # Stores the vertices and faces
    vertices, faces = [], []

    # Runs through each face
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):

                # Skips nodes without a face between them
                if not material.has_face(n1, n2, n3):
                    continue

                # Stores the used nodes
                used_nodes.add(n1)
                used_nodes.add(n2)
                used_nodes.add(n3)

                # Stores the used edges
                used_edges.add((n1,n2))
                used_edges.add((n1,n3))
                used_edges.add((n2,n3))

                # Computes the face's components
                face_vertices, face_faces = generate_face_surface_mesh(material, n1, n2, n3)

                # Stores the face's components
                vertices.append(face_vertices)
                faces.append(face_faces)

    # Runs through each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips nodes without an edge between them
            if (n1,n2) not in used_edges and not material.has_edge(n1, n2):
                continue

            # Stores the nodes
            used_nodes.add(n1)
            used_nodes.add(n2)

            # Computes the edge's components
            edge_vertices, edge_faces = generate_edge_surface_mesh(material, n1, n2)

            # Stores the edge's components
            vertices.append(edge_vertices)
            faces.append(edge_faces)

    # Runs through each node
    for node in used_nodes:

        # Computes the node's components
        node_vertices, node_faces = generate_node_surface_mesh(material, node)

        # Stores the node's components
        vertices.append(node_vertices)
        faces.append(node_faces)

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


def union_obj_components(vertices: list[list[tuple]], faces: list[list[tuple]], filepath: str, check_manifold=True, verbose=True):
    """
    Takes the union of all the components in the given source .obj
    file, and stores the union in the given output file.

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
        The filepath into which the unioned components will be
        stored.

    check_manifold: `bool`, optional
        Whether the resulting mesh will be checked to be a manifold.
        Is an expensive operation. Default is `True`.

    verbose: `bool`, optional
        Whether status messages will be printed to the console.
        Default is `True`.
    """

    # Prints a beginning status
    if verbose:
        print(f"Taking the union of {filepath}...")

    # Stores each component's mesh
    component_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces[i]) for i in range(len(vertices))]

    # Computes the union of the meshes
    union_mesh: trimesh.Trimesh = trimesh.boolean.union(component_meshes)

    # Checks for manifold properties
    if check_manifold and not union_mesh.is_watertight:
        print(f"WARNING: {filepath} is not a manifold mesh.")

    # Exports the mesh
    union_mesh.export(filepath)

    # Prints a ending status
    if verbose:
        print("Union finished!")
