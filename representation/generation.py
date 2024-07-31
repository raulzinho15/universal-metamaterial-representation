import numpy as np
import matplotlib.pyplot as plt
from random import random, choice

from representation.rep_class import *
from representation.rep_utils import *
from representation.utils import *
from autoencoder.autoencoder import *


FRAMES_PER_STEP = 10


def random_metamaterial(edge_prob=0.5, face_prob=0.5, grid_spacing=None, connected=False, cyclic=False, wavy_edges=False, validate=False):
    """
    Generates a random metamaterial's representation with its node positions,
    edge relations, and face relations. Implicitly determinable node positions
    are not explicitly included in the node position array since they are
    constant across all metamaterials.

    edge_prob: float
        The probability that any two nodes will have an edge between them,
        before any modifications for validity.

    face_prob: float
        The probability that any three nodes will have a face between them,
        before any modifications for validity.

    grid_spacing: int or None
        If an int, will only place nodes at random points along an evenly-spaced grid
        of the unit cube, where the number of grid spaces along each dimension
        is this value. Minimum of 1. If None, chooses random node positions.

    connected: bool
        Whether the metamaterial will be connected (no floating edge islands).

    cyclic: bool
        Whether the metamaterial will have no hanging connected edges/faces.

    validate: bool
        Whether to remove any invalid edges/faces from the generated metamaterial.

    wavy_edges: bool
        Whether to allow for wavy edges.
    
    Returns: Metamaterial
        A randomly generated metamaterial.
    """
    
    # Generates the node position representation array
    if grid_spacing is None or grid_spacing < 1:
        node_pos = np.random.rand(NODE_POS_SIZE)
    else:
        nums = [2*x/grid_spacing-1 for x in range(grid_spacing+1)]
        node_pos = np.array([euclidean_to_spherical(choice(nums),choice(nums),choice(nums)) for _ in range(NUM_NODES-1)]).reshape(NODE_POS_SIZE)

    # Generates the edge/face adjacency representation array
    edge_adj = np.zeros(EDGE_ADJ_SIZE)
    face_adj = np.zeros(FACE_ADJ_SIZE)
    while not edge_adj.any() and not face_adj.any():
        edge_adj = (np.random.rand(EDGE_ADJ_SIZE) < edge_prob).astype(float)
        face_adj = (np.random.rand(FACE_ADJ_SIZE) < face_prob).astype(float)

    # Generates the edge parameters representation array
    if wavy_edges:
        edge_params = np.random.randn(EDGE_PARAMS_SIZE)/5
    else:
        edge_params = np.zeros(EDGE_PARAMS_SIZE)
    
    metamaterial = Metamaterial(node_pos, edge_adj, edge_params, face_adj)

    if cyclic:
        metamaterial.remove_disconnections()
        metamaterial.remove_acycles()

    elif connected:
        metamaterial.remove_disconnections()

    # Ensures the representation is of a validly constructed metamaterial
    if validate:
        metamaterial.remove_invalid_faces() # Removes faces without all edges in the rep
        metamaterial.remove_invalid_edges() # Removes edges intersecting with faces
        metamaterial.remove_invalid_faces() # Removes faces without all edges in the rep after edge removal

    return metamaterial


# Computes consistent colors for each node
NODE_COLORS = {}
for n1 in range(NUM_NODES):
    NODE_COLORS[n1] = tuple(random() for _ in range(3))

# Computes consistent colors for each edge
EDGE_COLORS = {}
for n1 in range(NUM_NODES):
    for n2 in range(n1+1, NUM_NODES):
        EDGE_COLORS[(n1, n2)] = tuple(random() for _ in range(3))

# Computes consistent colors for each face
FACE_COLORS = {}
for n1 in range(NUM_NODES):
    for n2 in range(n1+1, NUM_NODES):
        for n3 in range(n2+1, NUM_NODES):
            FACE_COLORS[(n1, n2, n3)] = tuple(random() for _ in range(3))


def plot_metamaterial(metamaterial: Metamaterial, subplot=None, filename="", animate=False, plot_nodes=False):
    """
    Plots the metamaterial with the given representation at the given filename.

    metamaterial: Metamaterial
        The metamaterial to plot.

    subplot:
        The subplot into which the 3D figures will be drawn. If None, a local
        one will be created.

    filename: str
        The name of the file at which the plot image will be saved. If not
        specified, the file will not be saved.

    animate: bool
        Whether the rotating animation will be played.

    plot_nodes: bool
        Whether the nodes will be visually plotted.
    """
    
    # Sets up the 3d plot environment
    close = False
    if not subplot:
        close = True
        subplot = plt.figure().add_subplot(projection="3d")
        subplot.set_xlabel("x")
        subplot.set_ylabel("y")
        subplot.set_zlabel("z")

    # Plots each node
    if plot_nodes:
        for node in range(NUM_NODES):
            x,y,z = metamaterial.get_node_position(node)
            subplot.plot([x+0.1,x-0.1], [y+0.1, y-0.1], zs=[z+0.1, z-0.1], linewidth=5, color=NODE_COLORS[node])

    # Plots each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips unconnected nodes
            if not metamaterial.has_edge(n1, n2):
                continue

            # Computes the edge coordinates
            edge_points = metamaterial.compute_edge_points(n1, n2)

            # Plots the edge coordinates
            for i in range(len(edge_points)-1):

                # Gets the correct coordinates
                x1, y1, z1 = edge_points[i,:]
                x2, y2, z2 = edge_points[i+1,:]

                # Plots the edge segment
                subplot.plot([x1, x2], [y1, y2], zs=[z1, z2], linewidth=5, color=EDGE_COLORS[(n1, n2)])

    # Plots each face
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):

                # Skips unconnected nodes
                if not metamaterial.has_face(n1, n2, n3):
                    continue

                # Computes the face coordinates
                x1, y1, z1 = metamaterial.get_node_position(n1)
                x2, y2, z2 = metamaterial.get_node_position(n2)
                x3, y3, z3 = metamaterial.get_node_position(n3)

                subplot.plot_trisurf([x1, x2, x3], [y1, y2, y3], [z1, z2, z3], alpha=0.8, color='y', triangles=[[0,1,2]])

    if filename:
        plt.savefig(filename)

    # Plays a rotation animation
    if animate:
        for angle in range(0, 360*10):
            subplot.view_init(30, angle)
            plt.draw()
            plt.pause(.002)

    if close:
        plt.close()


def metamaterial_grid(metamaterial: Metamaterial, shape=(1,1,1)):
    """
    Produces a grid of mirrored metamaterials.

    metamaterial: Metamaterial
        The metamaterial to be gridded.

    shape: tuple of ints
        The amount in the x, y, and z directions to grid the metamaterial.
        If no gridding (i.e., just the metamaterial by itself), the input
        should be (1,1,1).

    Returns: list of Metamaterials
        The list of the metamaterials in the grid of the given shape.
    """
    
    # Stores the metamaterials to be plotted
    materials = [metamaterial]

    # Computes the metamaterials along the x axis
    materials = [material.mirror(x=(dx%2 == 1)).translate(dx=dx).toggle_plane_display(x0=(dx != 0))
                    for material in materials
                        for dx in range(shape[0])]

    # Computes the metamaterials along the y axis
    materials = [material.mirror(y=(dy%2 == 1)).translate(dy=dy).toggle_plane_display(y0=(dy != 0))
                    for material in materials
                        for dy in range(shape[1])]

    # Computes the metamaterials along the z axis
    materials = [material.mirror(z=(dz%2 == 1)).translate(dz=dz).toggle_plane_display(z0=(dz != 0))
                    for material in materials
                        for dz in range(shape[2])]

    return materials


def plot_metamaterial_grid(metamaterial, shape, filename="", animate=False):
    """
    Plots the metamaterial grid of the given shape at the given filename.

    metamaterial: Metamaterial
        The metamaterial to plot.

    shape: tuple of ints
        The amount in the x, y, and z directions to grid the metamaterial.
        If no gridding (i.e., just the metamaterial by itself), the input
        should be (1,1,1).

    filename: str
        The name of the file at which the plot image will be saved. If not
        specified, the file will not be saved.

    animate: bool
        Whether the rotating animation will be played.
    """

    # Stores the gridded metamaterials to be plotted
    materials = metamaterial_grid(metamaterial, shape)

    # Prepares the subplot
    fig = plt.figure()
    subplot = fig.add_subplot(projection="3d")
    subplot.set_xlabel("x")
    subplot.set_ylabel("y")
    subplot.set_zlabel("z")

    # Plots each metamaterial
    for material in materials:
        plot_metamaterial(material, subplot=subplot)

    if filename:
        plt.savefig(filename)

    # Plays a rotation animation
    if animate:
        for angle in range(0, 360*10):
            subplot.view_init(30, angle)
            plt.draw()
            plt.pause(.002)

    plt.close()


def generate_interpolation_steps(interps: int) -> torch.Tensor:

    interpolation_steps = torch.abs(torch.randn((REP_SIZE, interps)))
    return interpolation_steps / interpolation_steps.sum(dim=1, keepdim=True)


def interpolate(model: MetamaterialAE, material1: Metamaterial, material2: Metamaterial, interps: int, validate=False, func="linear", use_ae=False, random_steps=False):
    """
    Generates the linear interpolation of the two materials.

    model: MetamaterialAE
        The model to use to interpolate. Can be None if use_ae is False.

    material1: Metamaterial
        The base material for interpolation.

    material2: Metamaterial
        The material to be interpolated into.

    interps: int
        The number of interpolations to compute after the starting metamaterial.

    validate: bool
        Whether to validate the interpolated metamaterials and remove
        any invalid edges/faces.

    func: str
        The function to use for interpolation. If "linear" then uses linear interpolation.
        If "spherical" then uses spherical integration. By default, uses linear.

    use_ae: bool
        Whether the given model should be used.

    random_steps: bool
        Whether to use random steps.

    Returns: list[Metamaterial]
        The list of interpolated metamaterials.
    """

    # Computes the latent representation of the two metamaterials
    if use_ae:
        m1_latent = model.encode(material1.flatten_rep(pad_dim=True))
        m2_latent = model.encode(material2.flatten_rep(pad_dim=True))
    else:
        m1_latent = material1.flatten_rep()
        m2_latent = material2.flatten_rep()

    # Stores values for interpolation
    alpha = torch.zeros(REP_SIZE)
    if random_steps:
        alpha_steps = generate_interpolation_steps(interps)
    else:
        alpha_steps = torch.ones((REP_SIZE, interps)) / interps

    # Runs through each interpolation
    materials = []
    for interp in range(interps):

        # Updates the value of alpha
        alpha = alpha + alpha_steps[:,interp]

        # Interpolates according to the given function
        if func == "spherical":
            omega = torch.acos(torch.sum(m1_latent * m2_latent) / (torch.norm(m1_latent, p=2) * torch.norm(m2_latent, p=2))).item()
            interpolated_value = (np.sin((1-alpha) * omega) * m1_latent + np.sin(alpha * omega) * m2_latent) / np.sin(omega)
        else:
            interpolated_value = m1_latent * (1-alpha) + m2_latent * alpha

        # Decodes the interpolation
        if use_ae:
            interpolated_value = model.decode(interpolated_value)

        materials.append(Metamaterial.from_tensor(interpolated_value, random_cutoffs=False))

        # Validates the decoded representation
        if validate:
            materials[-1].remove_invalid_faces() # Removes faces without all edges in the rep
            materials[-1].remove_invalid_edges() # Removes edges intersecting with faces
            materials[-1].remove_invalid_faces() # Removes faces without all edges in the rep after edge removal

    return [material1] + materials + [material2]


def minimize_node_distance(node_pos1: np.ndarray, node_pos2: np.ndarray) -> np.ndarray:

    # Stores the offset node positions
    minus_node_pos2 = node_pos2 - np.array([0,0,1] * (node_pos2.shape[0]//3))
    plus_node_pos2 = node_pos2 + np.array([0,0,1] * (node_pos2.shape[0]//3))

    # Computes the minimum distance phi's
    new_node_pos = node_pos2.copy()
    for i in range(2, node_pos2.shape[0], 3):

        # Keeps the minus position's phi
        if np.abs(minus_node_pos2[i]-node_pos1[i]) < min(np.abs(node_pos2[i]-node_pos1[i]), np.abs(plus_node_pos2[i]-node_pos1[i])):
            new_node_pos[i] = minus_node_pos2[i]

        # Keeps the plus position's phi
        elif np.abs(plus_node_pos2[i]-node_pos1[i]) < min(np.abs(node_pos2[i]-node_pos1[i]), np.abs(minus_node_pos2[i]-node_pos1[i])):
            new_node_pos[i] = plus_node_pos2[i]

    return new_node_pos


def rotate_material_edge_params(material1: Metamaterial, material2: Metamaterial, invert_angle=False) -> Metamaterial:
    """
    Rotates the edge parameters in material2 to be in the coordinates
    of the edge parameters in material1. Does mutate material2.
    """
    
    # Maps the edge parameters in the second material to the coordinates of the first
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            
            # Stores the ordering of nodes
            mat1_node1,mat1_node2 = sorted((n1, n2), key=lambda n: material1.node_ordering[n])
            mat2_node1,mat2_node2 = sorted((n1, n2), key=lambda n: material2.node_ordering[n])

            # Stores the material1 edge vector
            mat1_node1_pos = material1.get_node_position(mat1_node1, transform=False)
            mat1_node2_pos = material1.get_node_position(mat1_node2, transform=False)
            mat1_edge = mat1_node2_pos - mat1_node1_pos

            # Stores the material2 edge vector
            mat2_node1_pos = material2.get_node_position(mat2_node1, transform=False)
            mat2_node2_pos = material2.get_node_position(mat2_node2, transform=False)
            mat2_edge = mat2_node2_pos - mat2_node1_pos

            # Stores the axis of rotation
            rotation_axis = np.cross(mat2_edge, mat1_edge)
            rotation_axis_mag = np.linalg.norm(rotation_axis)
            if rotation_axis_mag < 1e-4: # Skips edges on the same axis
                continue
            rotation_axis /= rotation_axis_mag

            # Computes the edge length
            mat1_edge_len = np.linalg.norm(mat1_edge)
            mat2_edge_len = np.linalg.norm(mat2_edge) 

            # Stores the angle of rotation
            dot_prod = np.dot(mat1_edge, mat2_edge)
            rotation_angle = np.arccos(dot_prod / mat1_edge_len / mat2_edge_len) * (-1 if invert_angle else 1)

            # Scales the edge parameters
            edge_index = edge_adj_index(n1,n2) * EDGE_BEZIER_COORDS
            material2.edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] *= mat2_edge_len/mat1_edge_len

            # Rotates material2's edge parameters
            material2.rotate_edge_params(n1, n2, rotation_axis, rotation_angle)


# Computes values for alpha for interpolation
def alpha_gen(num):
    for i in range(num):
        yield i/(num-1)


def interpolate_part_changes(original_material: Metamaterial, material: Metamaterial, part_changes: list[tuple[int]], end_node_pos: np.ndarray, end_edge_params: np.ndarray) -> tuple[Metamaterial, list[Metamaterial]]:

    # Computes copies for non-mutation
    mid_material = material.copy()
    mid_material.node_pos = end_node_pos.copy()
    mid_material.edge_params = end_edge_params.copy()

    # Computes and stores the average node position and the edge parameters
    for change in part_changes:

        # Stores the edge change nodes
        n1,n2,_ = change

        # Computes the node positions
        first_node_pos = material.node_pos[n1*3:(n1+1)*3]
        second_node_pos = minimize_node_distance(first_node_pos, material.node_pos[n2*3:(n2+1)*3])

        # Computes the average
        avg_node_pos = (first_node_pos + second_node_pos) / 2

        # Stores the average
        mid_material.node_pos[n1*3:(n1+1)*3] = avg_node_pos
        mid_material.node_pos[n2*3:(n2+1)*3] = avg_node_pos

        # Zeros the edge parameters between these two nodes
        edge_index = edge_adj_index(n1,n2)
        mid_material.edge_params[edge_index*EDGE_BEZIER_COORDS : (edge_index+1)*EDGE_BEZIER_COORDS] = 0

    # Prepares values for interpolation
    start_vector = material.flatten_rep()
    mid_material.node_pos = minimize_node_distance(material.node_pos, mid_material.node_pos)
    end_vector = mid_material.flatten_rep()

    # Generates the interpolated materials
    materials: list[Metamaterial] = []
    for alpha in alpha_gen(FRAMES_PER_STEP+1):

        # Creates the interpolated material
        mat: Metamaterial = Metamaterial.from_tensor(start_vector*(1-alpha) + end_vector*alpha)

        # Rotates the edge parameters according to the change since the last material
        rotate_material_edge_params(original_material, mat, invert_angle=True)

        # Stores the interpolated material
        materials.append(mat)

    # Computes the final material with the edges changed
    final_material = mid_material.copy()
    for change in part_changes:

        # Handles an edge change
        n1,n2,is_removal = change
        final_material.edge_adj[edge_adj_index(n1,n2)] = 0 if is_removal else 1

    return final_material, materials


def smooth_interpolation(material1: Metamaterial, material2: Metamaterial) -> list[Metamaterial]:

    # Stores the materials' adjacency matrices
    edge_adj_matrix1 = material1.get_edge_adj_matrix().astype(np.int8)
    edge_adj_matrix2 = material2.get_edge_adj_matrix().astype(np.int8)

    # Finds an approximately best topology match between the two materials
    edge_adj_matrix1, swaps = greedy_topology_match(edge_adj_matrix1, edge_adj_matrix2)

    # Performs each of the same swaps on the starting metamaterial
    node_order = [i for i in range(NUM_NODES)]
    for n1,n2 in swaps:
        node_order[n1], node_order[n2] = node_order[n2], node_order[n1]
    material1 = material1.reorder_nodes(node_order)

    # Computes the relevant subgraph adjacency matrices
    removed_edge_adj = ((edge_adj_matrix1 - edge_adj_matrix2) == 1).astype(np.int8)
    added_edge_adj = ((edge_adj_matrix2 - edge_adj_matrix1) == 1).astype(np.int8)

    # Makes material copies to avoid unintended mutations
    start_material = material1.copy()
    material2 = material2.copy()

    # Rotates material2's edge parameters
    rotate_material_edge_params(material1, material2)

    # Finds each edge/face removal/addition
    part_changes = (
        [(n1,n2,False) for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES) if added_edge_adj[n1,n2]] +
        [(n1,n2,True) for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES) if removed_edge_adj[n1,n2]]
    )

    # Runs parallelizable part changes concurrently
    part_change_groups = []
    connected_nodes = set([i for i in range(material1.active_nodes())])
    while part_changes:

        # Stores values for the edge change groups
        changed_indices = []
        seen_nodes = set()

        # Runs through each edge change
        for i, change in enumerate(part_changes):

            # Stores the change parameters
            n1,n2,_ = change

            # Skips changes affecting already target nodes
            if n1 in seen_nodes or n2 in seen_nodes:
                continue

            # Skips currently disconnected nodes
            if n1 not in connected_nodes and n2 not in connected_nodes:
                continue

            # Keeps track of the affected nodes
            seen_nodes.add(n1)
            seen_nodes.add(n2)
            connected_nodes.add(n1)
            connected_nodes.add(n2)
            changed_indices.append(i)

        # Stores all changes in this group
        part_change_groups.append([part_changes[i] for i in changed_indices])
        for i in changed_indices[-1::-1]:
            part_changes.pop(i)

    # Computes properties about the aggregate node positions
    mat1_nodes = material1.active_nodes()
    mat2_nodes = material2.active_nodes()
    mat1_node_pos = material1.node_pos.copy()
    mat2_node_pos = material2.node_pos.copy()

    # Handles material 2 having more new nodes
    if mat1_nodes < mat2_nodes:

        # Copies new node positions
        mat1_node_pos[mat1_nodes*3:] = mat2_node_pos[mat1_nodes*3:]

        # Stores the updated node positions
        start_material.node_pos = mat1_node_pos

    # Handles material 1 having more new nodes
    else:

        # Copies new node positions
        mat2_node_pos[mat2_nodes*3:] = mat1_node_pos[mat2_nodes*3:]

        # Stores the updated node positions
        material2.node_pos = mat2_node_pos

    # Computes properties about the edge parameters
    mat1_edge_params = material1.edge_params.copy()
    mat2_edge_params = material2.edge_params.copy()

    # Runs through each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Handles material 1 uniquely having the edge
            if removed_edge_adj[n1,n2]:
                edge_index = edge_adj_index(n1,n2) * EDGE_BEZIER_COORDS
                mat2_edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = material1.get_edge_params(n1,n2)

            # Handles material 1 uniquely having the edge
            elif added_edge_adj[n1,n2]:
                edge_index = edge_adj_index(n1,n2) * EDGE_BEZIER_COORDS
                mat1_edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = material2.get_edge_params(n1,n2)

    # Stores the updated edge parameters
    start_material.edge_params = mat1_edge_params
    material2.edge_params = mat2_edge_params

    # Computes the intermediate node positions and edge/face parameters
    node_positions = np.stack([mat1_node_pos * (1-alpha) + mat2_node_pos * alpha for alpha in alpha_gen(len(part_change_groups)+2)], axis=0)[1:]
    edge_params = np.stack([mat1_edge_params * (1-alpha) + mat2_edge_params * alpha for alpha in alpha_gen(len(part_change_groups)+2)], axis=0)[1:]

    # Executes each edge change
    materials = []
    for changes in part_change_groups:
        start_material, next_materials = interpolate_part_changes(material1, start_material, changes, node_positions[0], edge_params[0])
        node_positions = node_positions[1:]
        edge_params = edge_params[1:]
        materials.extend(next_materials[:-1])

    # Interpolates everything else if no changes were made
    start_vector = start_material.flatten_rep()
    end_vector = material2.flatten_rep()
    for alpha in alpha_gen(FRAMES_PER_STEP):

        # Creates the interpolated material
        mat: Metamaterial = Metamaterial.from_tensor(start_vector*(1-alpha) + end_vector*alpha)

        # Rotates the edge parameters according to the change since the last material
        rotate_material_edge_params(material1, mat, invert_angle=True)

        # Stores the interpolated material
        materials.append(mat)

    return materials


    # Change bezier coordinates to be relative to the point along the straight edge/face
    # that the coordinates are currently at
    # This will ensure that edge/face parameters interpolate smoothly as node positions
    # move around, since the axis/plane on which the bezier coordinates will be based on
    # will rotate with the nodes

