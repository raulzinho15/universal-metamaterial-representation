import numpy as np
import matplotlib.pyplot as plt
from random import random, choice, shuffle

from representation.rep_class import *
from representation.rep_utils import *
from representation.utils import *
from autoencoder.autoencoder import *


FRAMES_PER_STEP = 40


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


def rotate_material_edge_params(material1: Metamaterial, material2: Metamaterial, edge_lengths: np.ndarray, invert_angle=False) -> Metamaterial:
    """
    Rotates the edge parameters in material2 to be in the coordinates
    of the edge parameters in material1. Does mutate material2.
    """
    
    # Maps the edge parameters in the second material to the coordinates of the first
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            
            # Stores the ordering of nodes
            mat1_node1,mat1_node2 = sorted((n1, n2))
            mat2_node1,mat2_node2 = sorted((n1, n2))

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
            material2.edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] *= mat2_edge_len / edge_lengths[edge_adj_index(n1,n2)]

            # Rotates material2's edge parameters
            material2.rotate_edge_params(n1, n2, rotation_axis, rotation_angle)


# Computes values for alpha for interpolation
def alpha_gen(num):
    for i in range(num):
        yield i/(num-1)


def interpolate_part_changes(original_material: Metamaterial, material: Metamaterial, part_changes: list[tuple[int|bool]], end_node_pos: np.ndarray, end_edge_params: np.ndarray, edge_lengths: np.ndarray) -> tuple[Metamaterial, list[Metamaterial]]:

    # Computes the pre-change middle material
    pre_change_material = material.copy()
    pre_change_material.node_pos = (material.node_pos + end_node_pos) / 2
    pre_change_material.edge_params = (material.edge_params + end_edge_params) / 2

    # Computes and stores the average node position and the edge parameters
    for change in part_changes:

        # Stores the edge change nodes
        if len(change) == 3:
            n1,n2,_ = change
            n3 = n2
        
        # Stores the face change nodes
        else:
            n1,n2,n3,_ = change

        # Computes the node positions
        first_node_pos = pre_change_material.node_pos[n1*3:(n1+1)*3]
        second_node_pos = minimize_node_distance(first_node_pos, pre_change_material.node_pos[n2*3:(n2+1)*3])
        third_node_pos = minimize_node_distance(first_node_pos, pre_change_material.node_pos[n3*3:(n3+1)*3])

        # Computes the average
        avg_node_pos = (first_node_pos + second_node_pos) / 2 
        if n3 != n2: # Adjusts the average for a face change
            avg_node_pos = (2*avg_node_pos + third_node_pos) / 3

        # Stores the average
        pre_change_material.node_pos[n1*3:(n1+1)*3] = avg_node_pos
        pre_change_material.node_pos[n2*3:(n2+1)*3] = avg_node_pos
        pre_change_material.node_pos[n3*3:(n3+1)*3] = avg_node_pos

        # Zeros the edge parameters between nodes
        for node1,node2 in [(n1,n2), (n2,n3), (n1,n3)]:
            edge_index = edge_adj_index(node1,node2)
            pre_change_material.edge_params[edge_index*EDGE_BEZIER_COORDS : (edge_index+1)*EDGE_BEZIER_COORDS] = 0

    # Prepares values for the first interpolation
    start_vector = material.flatten_rep()
    pre_change_material.node_pos = minimize_node_distance(material.node_pos, pre_change_material.node_pos)
    end_vector = pre_change_material.flatten_rep()

    # Generates the first interpolated materials
    materials: list[Metamaterial] = []
    for i,alpha in enumerate(alpha_gen(FRAMES_PER_STEP+1)):

        # Skips the final material which will also be made in the next interpolation
        if i == FRAMES_PER_STEP:
            break

        # Creates the interpolated material
        mat: Metamaterial = Metamaterial.from_tensor(start_vector*(1-alpha) + end_vector*alpha)

        # Rotates the edge parameters according to the change since the last material
        rotate_material_edge_params(original_material, mat, edge_lengths[i], invert_angle=True)

        # Stores the interpolated material
        materials.append(mat)

    # Computes the post-change middle material
    post_change_material = pre_change_material.copy()
    post_change_material

    # Computes the edge/face adjacency for the post-change material
    for change in part_changes:

        # Handles an edge change
        if len(change) == 3:
            n1,n2,is_removal = change
            post_change_material.edge_adj[edge_adj_index(n1,n2)] = 0 if is_removal else 1
            continue

        # Handles a face change
        n1,n2,n3,is_removal = change
        post_change_material.face_adj[face_adj_index(n1,n2,n3)] = 0 if is_removal else 1

    # Computes the final material
    final_material = post_change_material.copy()
    final_material.node_pos = end_node_pos.copy()
    final_material.edge_params = end_edge_params.copy()

    # Prepares values for the first interpolation
    start_vector = post_change_material.flatten_rep()
    old_final_node_pos = final_material.node_pos.copy()
    final_material.node_pos = minimize_node_distance(material.node_pos, final_material.node_pos)
    end_vector = final_material.flatten_rep()
    final_material.node_pos = old_final_node_pos

    # Generates the second interpolated materials
    for i,alpha in enumerate(alpha_gen(FRAMES_PER_STEP+1)):

        # Skips the final material which will also be made in the next interpolation
        if i == FRAMES_PER_STEP:
            break

        # Creates the interpolated material
        mat: Metamaterial = Metamaterial.from_tensor(start_vector*(1-alpha) + end_vector*alpha)

        # Rotates the edge parameters according to the change since the last material
        rotate_material_edge_params(original_material, mat, edge_lengths[FRAMES_PER_STEP+i], invert_angle=True)

        # Stores the interpolated material
        materials.append(mat)

    return final_material, materials


def smooth_interpolation(material1: Metamaterial, material2: Metamaterial) -> list[Metamaterial]:

    # Stores the materials' edge adjacency matrices
    edge_adj_matrix1 = material1.get_edge_adj_matrix().astype(np.int8)
    edge_adj_matrix2 = material2.get_edge_adj_matrix().astype(np.int8)

    # Finds an approximately best topology match between the two materials
    edge_adj_matrix1, swaps = greedy_topology_match(edge_adj_matrix1, edge_adj_matrix2)

    # Performs each of the same swaps on the starting metamaterial
    node_order = [i for i in range(NUM_NODES)]
    for n1,n2 in swaps:
        node_order[n1], node_order[n2] = node_order[n2], node_order[n1]
    material1 = material1.reorder_nodes(node_order)

    # Makes material copies to avoid unintended mutations
    material2 = material2.copy()

    # Computes the relevant subgraph edge adjacency matrices
    removed_edge_adj = ((edge_adj_matrix1 - edge_adj_matrix2) == 1).astype(np.int8)
    added_edge_adj = ((edge_adj_matrix2 - edge_adj_matrix1) == 1).astype(np.int8)

    # Stores the materials' face adjacency matrices
    face_adj_tensor1 = material1.get_face_adj_tensor().astype(np.int8)
    face_adj_tensor2 = material2.get_face_adj_tensor().astype(np.int8)

    # Computes the relevant subgraph face adjacency matrices
    removed_face_adj = ((face_adj_tensor1 - face_adj_tensor2) == 1).astype(np.int8)
    added_face_adj = ((face_adj_tensor2 - face_adj_tensor1) == 1).astype(np.int8)

    # Stores the active nodes in the materials
    mat1_nodes = material1.active_nodes()
    mat2_nodes = material2.active_nodes()

    # Stores all active nodes across all materials
    all_active_nodes = mat1_nodes.union(mat2_nodes)
    
    # Stores an out-of-bounds point
    out_of_bounds = euclidean_to_pseudo_spherical(NON_COLINEAR_POINT[np.newaxis,:])

    # Sets non-active nodes to an out-of-bounds point
    for node in range(NUM_NODES):

        # Skips nodes being used
        if node in all_active_nodes:
            continue

        # Marks the nodes as out-of-bounds
        material1.node_pos[node*3 : (node+1)*3] = out_of_bounds.copy()
        material2.node_pos[node*3 : (node+1)*3] = out_of_bounds.copy()

    # Finds each edge/face removal/addition
    part_changes = (
        [(n1,n2,n3,False)  for n1 in range(NUM_NODES)  for n2 in range(n1+1, NUM_NODES)  for n3 in range(n2+1, NUM_NODES) if added_face_adj[n1,n2,n3]  ] +
        [(n1,n2,False)     for n1 in range(NUM_NODES)  for n2 in range(n1+1, NUM_NODES)                                   if added_edge_adj[n1,n2]     ] +
        [(n1,n2,n3,True)   for n1 in range(NUM_NODES)  for n2 in range(n1+1, NUM_NODES)  for n3 in range(n2+1, NUM_NODES) if removed_face_adj[n1,n2,n3]] +
        [(n1,n2,True)      for n1 in range(NUM_NODES)  for n2 in range(n1+1, NUM_NODES)                                   if removed_edge_adj[n1,n2]   ]
    )

    # Runs parallelizable part changes concurrently
    part_change_groups = []
    connected_nodes = all_active_nodes.copy()
    while part_changes:

        # Stores values for the edge change groups
        changed_indices = []
        seen_nodes = set()

        # Runs through each edge change
        for i, change in enumerate(part_changes):

            # Stores the edge change parameters
            if len(change) == 3:
                n1,n2,is_removal = change
                n3 = n2
            
            # Stores the face change parameters
            else:
                n1,n2,n3,is_removal = change

            # Skips changes affecting already target nodes
            if n1 in seen_nodes or n2 in seen_nodes or n3 in seen_nodes:
                continue

            # Skips currently disconnected nodes
            if n1 not in connected_nodes and n2 not in connected_nodes and n3 not in connected_nodes:
                continue

            # Ignores an edge change if a dependent face has not yet called on it
            if n3 == n2:

                # Finds whether there is a future face depending on this edge change
                found = False
                for node3 in range(NUM_NODES):

                    # Skips invalid faces
                    if n1 == node3 or n2 == node3:
                        continue

                    # Changes if the face exists in the future
                    nn1,nn2,nn3 = sorted((n1,n2,node3))
                    if (nn1,nn2,nn3,is_removal) in part_changes:
                        found = True
                        break

                # Skips the edge if a future face was found
                if found:
                    continue

            # Keeps track of the seen nodes
            seen_nodes.add(n1)
            seen_nodes.add(n2)
            seen_nodes.add(n3)

            # Keeps track of the nodes connected to the main graph
            connected_nodes.add(n1)
            connected_nodes.add(n2)
            connected_nodes.add(n3)

            # Stores this change as a change to make in the correct order
            changed_indices.append(i)

            # Checks if this face has any edge change dependencies
            if n3 != n2:

                # Checks every possible edge change
                for node1,node2 in [(n1,n2), (n1,n3), (n2,n3)]:
                    edge_change = (node1, node2, is_removal)

                    # Runs through each existent edge change
                    if edge_change in part_changes:

                        # If the face is being added, immediately call
                        # on the edge change to avoid an edge-less face
                        if not is_removal:
                            changed_indices.append(part_changes.index(edge_change))
                            continue
                        
                        # Checks if a future face removal depends on this edge change
                        found = False
                        for node3 in range(NUM_NODES):

                            # Skips invalid faces
                            if node1 == node3 or node2 == node3:
                                continue
                            nn1,nn2,nn3 = sorted((node1,node2,node3))
                            face_change = (nn1,nn2,nn3,is_removal)
                            if face_change == change:
                                continue

                            # Marks that a future face change was found
                            if face_change in part_changes:
                                found = True
                                break

                        # If there are not any future face changes dependent on this
                        # edge change, then let this face change call on the edge change
                        # so that no hanging edges from a former face are left
                        if not found:
                            changed_indices.append(part_changes.index(edge_change))

        # Stores all changes in this group
        changed_indices = sorted(changed_indices)
        part_change_groups.append([part_changes[i] for i in changed_indices])
        for i in changed_indices[-1::-1]:
            part_changes.pop(i)

    # Stores the materials' node positions
    mat1_node_pos = material1.node_pos.copy()
    mat2_node_pos = material2.node_pos.copy()

    # Copies node positions from other material for unused nodes
    for node in range(NUM_NODES):

        # Handles material1 having the node but not material2
        if node in mat1_nodes and node not in mat2_nodes:
            mat2_node_pos[node*3 : (node+1)*3] = mat1_node_pos[node*3 : (node+1)*3]

        # Handles material2 having the node but not material1
        elif node not in mat1_nodes and node in mat2_nodes:
            mat1_node_pos[node*3 : (node+1)*3] = mat2_node_pos[node*3 : (node+1)*3]
            
    # Stores the updated node positions
    material1.node_pos = mat1_node_pos
    material2.node_pos = mat2_node_pos

    # Stores each material's node positions in Euclidean space
    mat1_euclidean = pseudo_spherical_to_euclidean(mat1_node_pos.reshape((-1,3)))
    mat2_euclidean = pseudo_spherical_to_euclidean(mat2_node_pos.reshape((-1,3)))

    # Stores material1's edge lengths
    mat1_edge_lengths = np.array([
        np.sqrt( ((mat1_euclidean[n1] - mat1_euclidean[n2])**2).sum() )
            for n1 in range(NUM_NODES)
                for n2 in range(n1+1, NUM_NODES)
    ])

    # Stores material2's edge lengths
    mat2_edge_lengths = np.array([
        np.sqrt( ((mat2_euclidean[n1] - mat2_euclidean[n2])**2).sum() )
            for n1 in range(NUM_NODES)
                for n2 in range(n1+1, NUM_NODES)
    ])

    # Stores the interpolated edge lengths
    edge_lengths = np.concatenate([
        [mat1_edge_lengths * (1-alpha) + mat2_edge_lengths * alpha]
            for alpha in alpha_gen((2*len(part_change_groups)+1)*FRAMES_PER_STEP)
    ], axis=0)

    # Rotates material2's edge parameters
    rotate_material_edge_params(material1, material2, edge_lengths[-1])

    # Computes properties about the edge parameters
    mat1_edge_params = material1.edge_params.copy()
    mat2_edge_params = material2.edge_params.copy()

    # Runs through each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Handles material1 uniquely having the edge
            if removed_edge_adj[n1,n2]:
                edge_index = edge_adj_index(n1,n2) * EDGE_BEZIER_COORDS
                mat2_edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = material1.get_edge_params(n1,n2)

            # Handles material2 uniquely having the edge
            elif added_edge_adj[n1,n2]:
                edge_index = edge_adj_index(n1,n2) * EDGE_BEZIER_COORDS
                mat1_edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = material2.get_edge_params(n1,n2)

    # Stores the updated edge parameters
    material1.edge_params = mat1_edge_params
    material2.edge_params = mat2_edge_params

    # Stores the starting material
    start_material = material1.copy()

    # Computes the intermediate node positions and edge/face parameters
    node_positions = np.stack([mat1_node_pos * (1-alpha) + mat2_node_pos * alpha for alpha in alpha_gen(len(part_change_groups)+2)], axis=0)[1:]
    edge_params = np.stack([mat1_edge_params * (1-alpha) + mat2_edge_params * alpha for alpha in alpha_gen(len(part_change_groups)+2)], axis=0)[1:]

    # Executes each edge change
    materials = []
    for i,changes in enumerate(part_change_groups):

        # Stores the edge lengths to use for this interpolation
        next_edge_lengths = edge_lengths[:2*FRAMES_PER_STEP]

        # Computes the interpolation
        start_material, next_materials = interpolate_part_changes(material1, start_material, changes, node_positions[0], edge_params[0], next_edge_lengths)

        # Moves on to the next target properties
        node_positions = node_positions[1:]
        edge_params = edge_params[1:]
        edge_lengths = edge_lengths[2*FRAMES_PER_STEP:]

        # Stores the interpolated materials
        materials.extend(next_materials)

    # Computes the final, non-edge/face changing interpolation
    start_vector = start_material.flatten_rep()
    end_vector = material2.flatten_rep()
    for i,alpha in enumerate(alpha_gen(FRAMES_PER_STEP)):

        # Creates the interpolated material
        mat: Metamaterial = Metamaterial.from_tensor(start_vector*(1-alpha) + end_vector*alpha)

        # Rotates the edge parameters
        rotate_material_edge_params(material1, mat, edge_lengths[i], invert_angle=True)

        # Stores the interpolated material
        materials.append(mat)

    return materials


def baseline_interpolation(material1: Metamaterial, material2: Metamaterial, steps: int) -> list[Metamaterial]:

    # Stores the materials' edge adjacency matrices
    edge_adj1 = material1.get_edge_adj_matrix()
    edge_adj2 = material2.get_edge_adj_matrix()

    # Stores the materials' face adjacency tensors
    face_adj1 = material1.get_face_adj_tensor()
    face_adj2 = material2.get_face_adj_tensor()

    # Finds all the edge/face changes
    edge_changes = np.abs(edge_adj1-edge_adj2) == 1
    face_changes = np.abs(face_adj1-face_adj2) == 1

    # Accumulates all the changes
    part_changes = (
        [(n1,n2)    for n1 in range(NUM_NODES)  for n2 in range(n1+1, NUM_NODES)                                    if edge_changes[n1,n2]   ] +
        [(n1,n2,n3) for n1 in range(NUM_NODES)  for n2 in range(n1+1, NUM_NODES)  for n3 in range(n2+1, NUM_NODES)  if face_changes[n1,n2,n3]] 
    )
    shuffle(part_changes)

    # Stores the edge/face cutoff thresholds
    thresholds = [((i+1) / len(part_changes) - 1e-4) for i in range(len(part_changes))]

    # Stores the vectors for interpolation
    start_vector = material1.flatten_rep()
    end_vector = material2.flatten_rep()

    # Runs through the interpolation
    materials: list[Metamaterial] = []
    for alpha in alpha_gen(steps):

        # Computes the interpolated material
        material: Metamaterial = Metamaterial.from_tensor(start_vector * (1-alpha) + end_vector * alpha)

        # Updates the edge/face adjacencies according to the custom cutoffs
        for threshold,change in zip(thresholds, part_changes):

            # Handles setting the threshold for an edge change
            if len(change) == 2:
                edge_index = edge_adj_index(*change)
                material.edge_adj[edge_index] = material2.has_edge(*change) if alpha > threshold else material1.has_edge(*change)
            
            # Handles setting the threshold for a face change
            else:
                face_index = face_adj_index(*change)
                material.edge_adj[face_index] = material2.has_face(*change) if alpha > threshold else material1.has_face(*change)

        # Stores the interpolated material
        materials.append(material)

    return materials


