import numpy as np
import matplotlib.pyplot as plt
from random import random, choice

from representation.rep_class import *
from representation.rep_utils import *
from autoencoder.autoencoder import *


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
        node_pos = np.array([euclidian_to_spherical(choice(nums),choice(nums),choice(nums)) for _ in range(NUM_NODES-1)]).reshape(NODE_POS_SIZE)

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


def metamaterial_grid(metamaterial, shape):
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
    new_materials = []
    for dx in range(shape[0]):
        mirror = dx % 2 == 1
        new_materials += [material.mirror(x=mirror).translate(dx=dx) for material in materials]
    materials = new_materials

    # Computes the metamaterials along the y axis
    new_materials = []
    for dy in range(shape[1]):
        mirror = dy % 2 == 1
        new_materials += [material.mirror(y=mirror).translate(dy=dy) for material in materials]
    materials = new_materials

    # Computes the metamaterials along the z axis
    new_materials = []
    for dz in range(shape[2]):
        mirror = dz % 2 == 1
        new_materials += [material.mirror(z=mirror).translate(dz=dz) for material in materials]

    return new_materials


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


def interpolate(model: MetamaterialAE, material1: Metamaterial, material2: Metamaterial, interps, validate=False, func="linear"):
    """
    Generates the linear interpolation of the two materials.

    model: MetamaterialAE
        The model to use to interpolate.

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

    Returns: list[Metamaterial]
        The list of interpolated metamaterials.
    """

    # Computes the latent representation of the two metamaterials
    m1_latent = model.encoder(material1.flatten_rep(pad_dim=True))
    m2_latent = model.encoder(material2.flatten_rep(pad_dim=True))

    # Runs through each interpolation
    materials = []
    for ind, alpha in enumerate([x/interps for x in range(interps+1)]):

        # Decodes the interpolated latent representation
        if ind == 0:
            materials.append(material1)
        elif ind == interps:
            materials.append(material2)
        else:

            # Interpolates according to the given function
            if func == "spherical":
                omega = torch.acos(torch.sum(m1_latent * m2_latent) / (torch.norm(m1_latent, p=2) * torch.norm(m2_latent, p=2))).item()
                decoding = model.decoder((np.sin((1-alpha) * omega) * m1_latent + np.sin(alpha * omega) * m2_latent) / np.sin(omega))
            else:
                decoding = model.decoder(m1_latent*(1-alpha) + m2_latent*alpha)

            materials.append(Metamaterial.from_tensor(decoding))

        # Validates the decoded representation
        if validate:
            materials[-1].remove_invalid_faces() # Removes faces without all edges in the rep
            materials[-1].remove_invalid_edges() # Removes edges intersecting with faces
            materials[-1].remove_invalid_faces() # Removes faces without all edges in the rep after edge removal

    return materials


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