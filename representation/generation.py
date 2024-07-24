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


def plot_interpolation(model, material1: Metamaterial, material2: Metamaterial, interps, path, validate=False, shape=(1,1,1), func="linear"):
    """
    Plots the interpolation between the two given materials.

    model: Metamaterial
        The model to use to interpolate.

    material1: Metamaterial
        The base material for interpolation.

    material2: Metamaterial
        The material to be interpolated into.

    interps: int
        The number of interpolations to compute after the starting metamaterial.

    path: str
        The path at which the intermediate metamaterials will be placed.

    validate: bool
        Whether to validate the interpolated metamaterials and remove
        any invalid edges/faces.

    shape: tuple
        The gridded shape of the plotting. If (1,1,1), it is just the metamaterial by itself.

    func: str
        The function to use for interpolation. If "linear" then uses linear interpolation.
        If "spherical" then uses spherical integration. By default, uses linear.
    """

    for ind, material in enumerate(interpolate(model, material1, material2, interps, validate=validate, func=func)):
        plot_metamaterial_grid(material, shape=shape, filename=f"{path}/metamaterial{ind}.png")
        # plot_metamaterial_grid(material, shape=shape, filename=f"{path}/metamaterial{interps*2-ind}.png")


def align_nodes(mat1: Metamaterial, mat2: Metamaterial, nodes1, nodes2):
    """
        Aligns the two metamaterials in such a way that minimizes the collective
        distance of their nodes. Does not mutate the metamaterials.

        mat1: Metamaterial
            The first metamaterial for alignment.

        mat2: Metamaterial
            The second metamaterial for alignment.

        nodes1: int
            The number of nodes from mat1 (from index 0) that will be compared
            in the matching. Must be <= NUM_NODES and <= nodes2.

        nodes2: int
            The number of nodes from mat2 (from index 0) that will be compared
            in the matching. Must be <= NUM_NODES and >= nodes1.

        Returns: tuple of Metamaterials
            The aligned pair of metamaterials.
    """

    # Stores the reordering and includes leftover nodes
    reordering = mat1.best_node_match(mat2, nodes1, nodes2)
    reordering += [i for i in range(NUM_NODES) if i not in reordering]

    # (Forcefully) aligns the closest nodes 
    mat2 = mat2.reorder_nodes(reordering)
    mat1.node_pos[2*nodes1:] = mat2.node_pos[2*nodes1:]

    return mat1, mat2


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


# Computes values for alpha for interpolation
def alpha_gen(num):
    for i in range(num):
        yield i/(num-1)


def interpolate_part_changes(material: Metamaterial, part_changes: list[tuple[int]], end_node_pos: np.ndarray, end_edge_params: np.ndarray, end_face_params: np.ndarray) -> tuple[Metamaterial, list[Metamaterial]]:

    # Stores the interpolated materials
    materials = []

    # Computes copies for non-mutation
    mid1_material = material.copy()
    mid1_material.node_pos = end_node_pos.copy()
    mid1_material.edge_params = end_edge_params.copy()
    mid1_material.face_params = end_face_params.copy()

    # Computes and stores the average node position
    for change in part_changes:

        # Stores the edge change nodes
        if len(change) == 3:
            n1,n2,_ = change
            n3 = n2

        # Stores the face change nodes
        else:
            n1,n2,n3,_ = change

        # Computes the node positions
        first_node_pos = material.node_pos[n1*3:(n1+1)*3]
        second_node_pos = minimize_node_distance(first_node_pos, material.node_pos[n2*3:(n2+1)*3])

        # Computes the average
        avg_node_pos = (first_node_pos + second_node_pos) / 2
        if n2 == n3:
            third_node_pos = minimize_node_distance(first_node_pos, material.node_pos[n3*3:(n3+1)*3])
            avg_node_pos = (2*avg_node_pos + third_node_pos) / 3

        # Stores the average
        mid1_material.node_pos[n1*3:(n1+1)*3] = avg_node_pos
        mid1_material.node_pos[n2*3:(n2+1)*3] = avg_node_pos
        mid1_material.node_pos[n3*3:(n3+1)*3] = avg_node_pos

    # Computes the middle material with the edges/faces changed
    mid2_material = mid1_material.copy()
    for change in part_changes:

        # Handles an edge change
        if len(change) == 3:
            n1,n2,is_removal = change
            mid2_material.edge_adj[edge_adj_index(n1,n2)] = 0 if is_removal else 1
            continue

        # Handles a face change
        n1,n2,n3,is_removal = change
        mid2_material.face_adj[face_adj_index(n1,n2,n3)] = 0 if is_removal else 1

    # Stores the first half of the interpolated materials
    start_vector = material.flatten_rep()
    mid1_material.node_pos = minimize_node_distance(material.node_pos, mid1_material.node_pos)
    end_vector = mid1_material.flatten_rep()
    materials.extend(Metamaterial.from_tensor(start_vector*(1-alpha) + end_vector*alpha) for alpha in alpha_gen(FRAMES_PER_STEP+1))

    return mid2_material, materials


def smooth_interpolation(material1: Metamaterial, material2: Metamaterial) -> list[Metamaterial]:

    # Stores the materials' adjacency matrices
    edge_adj_matrix1 = material1.get_edge_adj_matrix().astype(np.int8)
    edge_adj_matrix2 = material2.get_edge_adj_matrix().astype(np.int8)
    face_adj_tensor1 = material1.get_face_adj_tensor().astype(np.int8)
    face_adj_tensor2 = material2.get_face_adj_tensor().astype(np.int8)

    # Finds the best topology match between the two materials
    edge_adj_matrix1, swaps = greedy_topology_match(edge_adj_matrix1, edge_adj_matrix2) ######### MAKE THIS ALSO SWAP FACE ADJACENCY

    # Performs each of the same swaps on the starting metamaterial
    for n1,n2 in swaps:
        node_order = [i for i in range(NUM_NODES)]
        node_order[n1] = n2
        node_order[n2] = n1
        material1 = material1.reorder_nodes(node_order)

    # Computes the relevant subgraph adjacency matrices
    removed_edge_adj = ((edge_adj_matrix1 - edge_adj_matrix2) == 1).astype(np.int8)
    added_edge_adj = ((edge_adj_matrix2 - edge_adj_matrix1) == 1).astype(np.int8)
    removed_face_adj = ((face_adj_tensor1 - face_adj_tensor2) == 1).astype(np.int8)
    added_face_adj = ((face_adj_tensor2 - face_adj_tensor1) == 1).astype(np.int8)

    # Stores the interpolated materials
    start_material = material1.copy()
    material2 = material2.copy()
    materials = []

    # Finds each edge/face removal/addition
    part_changes = (
        [(n1,n2,False) for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES) if added_edge_adj[n1,n2] == 1] +
        [(n1,n2,n3,False) for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES) for n3 in range(n2+1, NUM_NODES) if added_face_adj[n1,n2,n3] == 1] +
        [(n1,n2,True) for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES) if removed_edge_adj[n1,n2] == 1] +
        [(n1,n2,n3,True) for n1 in range(NUM_NODES) for n2 in range(n1+1, NUM_NODES) for n3 in range(n2+1, NUM_NODES) if removed_face_adj[n1,n2,n3] == 1]
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
            if len(change) == 3: # Edge change
                n1,n2,_ = change
                n3 = n1
            else: # Face change
                n1,n2,n3,_ = change

            # Skips changes affecting already target nodes
            if n1 in seen_nodes or n2 in seen_nodes or n3 in seen_nodes:
                continue

            # Skips currently disconnected nodes
            if n1 not in connected_nodes and n2 not in connected_nodes and n3 not in connected_nodes:
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

    # Computes properties about the aggregate node positions and edge/face parameters
    mat1_nodes = material1.active_nodes()
    mat2_nodes = material2.active_nodes()
    mat1_node_pos = material1.node_pos.copy()
    mat2_node_pos = material2.node_pos.copy()
    mat1_edge_params = material1.edge_params.copy()
    mat2_edge_params = material2.edge_params.copy()
    mat1_face_params = material1.face_params.copy()
    mat2_face_params = material2.face_params.copy()

    # Handles material 2 having more new nodes
    if mat1_nodes < mat2_nodes:

        # Copies new node positions
        mat1_node_pos[mat1_nodes*3:] = mat2_node_pos[mat1_nodes*3:]

        # Copies new edge parameters
        for n1 in range(mat1_nodes, NUM_NODES):
            for n2 in range(n1):
                edge_index = edge_adj_index(n1,n2) * EDGE_BEZIER_COORDS
                mat1_edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = material2.get_edge_params(n1,n2)

        # Copies new face parameters
        for n1 in range(mat1_nodes, NUM_NODES):
            for n2 in range(n1):
                for n3 in range(n2):
                    face_index = face_adj_index(n1,n2,n3) * FACE_BEZIER_COORDS
                    mat1_face_params[face_index : face_index + FACE_BEZIER_COORDS] = material2.get_face_params(n1,n2,n3)

        # Stores the updated node positions and edge/face parameters
        start_material.node_pos = mat1_node_pos
        start_material.edge_params = mat1_edge_params
        start_material.face_params = mat1_face_params

    # Handles material 1 having more new nodes
    else:

        # Copies new node positions
        mat2_node_pos[mat2_nodes*3:] = mat1_node_pos[mat2_nodes*3:]

        # Copies new edge parameters
        for n1 in range(mat2_nodes, NUM_NODES):
            for n2 in range(n1):
                edge_index = edge_adj_index(n1,n2) * EDGE_BEZIER_COORDS
                mat2_edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = material1.get_edge_params(n1,n2)

        # Copies new face parameters
        for n1 in range(mat2_nodes, NUM_NODES):
            for n2 in range(n1):
                for n3 in range(n2):
                    face_index = face_adj_index(n1,n2,n3) * FACE_BEZIER_COORDS
                    mat2_face_params[face_index : face_index + FACE_BEZIER_COORDS] = material1.get_face_params(n1,n2,n3)

        # Stores the updated node positions and edge/face parameters
        material2.node_pos = mat2_node_pos
        material2.edge_params = mat2_edge_params
        material2.face_params = mat2_face_params

    # Computes the intermediate node positions and edge/face parameters
    node_positions = np.stack([mat1_node_pos * (1-alpha) + mat2_node_pos * alpha for alpha in alpha_gen(len(part_change_groups)+2)], axis=0)[1:]
    edge_params = np.stack([mat1_edge_params * (1-alpha) + mat2_edge_params * alpha for alpha in alpha_gen(len(part_change_groups)+2)], axis=0)[1:]
    face_params = np.stack([mat1_face_params * (1-alpha) + mat2_face_params * alpha for alpha in alpha_gen(len(part_change_groups)+2)], axis=0)[1:]

    # Executes each edge change
    for changes in part_change_groups:
        start_material, next_materials = interpolate_part_changes(start_material, changes, node_positions[0], edge_params[0], face_params[0])
        node_positions = node_positions[1:]
        edge_params = edge_params[1:]
        face_params = face_params[1:]
        materials.extend(next_materials[:-1])

    # Interpolates everything else if no changes were made
    start_vector = start_material.flatten_rep()
    end_vector = material2.flatten_rep()
    materials.extend(Metamaterial.from_tensor(start_vector*(1-alpha) + end_vector*alpha) for alpha in alpha_gen(FRAMES_PER_STEP))

    return materials


    # Change bezier coordinates to be relative to the point along the straight edge/face
    # that the coordinates are currently at
    # This will ensure that edge/face parameters interpolate smoothly as node positions
    # move around, since the axis/plane on which the bezier coordinates will be based on
    # will rotate with the nodes

