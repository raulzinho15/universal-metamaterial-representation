import numpy as np
import matplotlib.pyplot as plt
from random import random

from representation.rep_class import *
from representation.rep_utils import *


def random_metamaterial(edge_prob=0.5, face_prob=0.5, with_faces=True, validate=False):
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

    with_faces: bool
        Whether the metamaterial will be generated with faces or not.

    validate: bool
        Whether to remove any invalid edges/faces from the generated metamaterial.
    
    Returns: Metamaterial
        A randomly generated metamaterial.
    """
    
    # Generates the node position representation array
    node_pos = np.random.rand(NODE_POS_SIZE)

    # Generates the edge adjacency representation array
    edge_adj = (np.random.rand(EDGE_ADJ_SIZE) < edge_prob).astype(float)
    
    # Generates the face adjacency representation array
    if with_faces:
        face_adj = (np.random.rand(FACE_ADJ_SIZE) < face_prob).astype(float)
    else:
        face_adj = np.zeros(FACE_ADJ_SIZE)

    metamaterial = Metamaterial(node_pos, edge_adj, face_adj)

    # Ensures the representation is of a validly constructed metamaterial
    if validate:
        metamaterial.remove_invalid_faces() # Removes faces without all edges in the rep
        metamaterial.remove_invalid_edges() # Removes edges intersecting with faces
        metamaterial.remove_invalid_faces() # Removes faces without all edges in the rep after edge removal

    return metamaterial

# Computes consistent colors for each edge
colors = {}
for n1 in range(NUM_NODES):
    for n2 in range(n1+1, NUM_NODES):
        colors[(n1, n2)] = tuple(random() for _ in range(3))

def plot_metamaterial(metamaterial, subplot=None, filename="", animate=False, save=False):
    """
    Plots the metamaterial with the given representation at the given filename.

    metamaterial: Metamaterial
        The metamaterial to plot.

    subplot:
        The subplot into which the 3D figures will be drawn.

    filename: str
        The name of the file at which the plot image will be saved.

    animate: bool
        Whether the rotating animation will be played.

    save: bool
        Whether the plot will be saved as an image file.
    """
    
    # Sets up the 3d plot environment
    close = False
    if not subplot:
        close = True
        subplot = plt.figure().add_subplot(projection="3d")
        subplot.set_xlabel("x")
        subplot.set_ylabel("y")
        subplot.set_zlabel("z")

    # Plots each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips unconnected nodes
            if not metamaterial.have_edge(n1, n2):
                continue

            # Computes the edge coordinates
            x1, y1, z1 = metamaterial.get_node_position(n1)
            x2, y2, z2 = metamaterial.get_node_position(n2)

            subplot.plot([x1, x2], [y1, y2], zs=[z1, z2], linewidth=5, color=colors[(n1, n2)])

    # Plots each face
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):
            for n3 in range(n2+1, NUM_NODES):

                # Skips unconnected nodes
                if not metamaterial.have_face(n1, n2, n3):
                    continue

                # Computes the face coordinates
                x1, y1, z1 = metamaterial.get_node_position(n1)
                x2, y2, z2 = metamaterial.get_node_position(n2)
                x3, y3, z3 = metamaterial.get_node_position(n3)

                subplot.plot_trisurf([x1, x2, x3], [y1, y2, y3], [z1, z2, z3], alpha=0.4)

    if save:
        plt.savefig(filename)

    # Plays a rotation animation
    if animate:
        for angle in range(0, 360*10):
            subplot.view_init(30, angle)
            plt.draw()
            plt.pause(.002)

    if close:
        plt.close()


def plot_metamaterial_grid(metamaterial, shape, filename="", save=False, animate=False):
    
    # Stores the metamaterials to be plotted
    materials = [metamaterial]

    # Computes the metamaterials along the x axis
    new_materials = []
    for dx in range(0, shape[0]):
        mirror = dx % 2 == 1
        new_materials += [material.mirror(x=mirror).translate(dx=dx) for material in materials]
    materials = new_materials

    # Computes the metamaterials along the y axis
    new_materials = []
    for dy in range(0, shape[1]):
        mirror = dy % 2 == 1
        new_materials += [material.mirror(y=mirror).translate(dy=dy) for material in materials]
    materials = new_materials

    # Computes the metamaterials along the z axis
    new_materials = []
    for dz in range(0, shape[2]):
        mirror = dz % 2 == 1
        new_materials += [material.mirror(z=mirror).translate(dz=dz) for material in materials]
    materials = new_materials

    # Prepares the subplot
    fig = plt.figure()
    subplot = fig.add_subplot(projection="3d")
    subplot.set_xlabel("x")
    subplot.set_ylabel("y")
    subplot.set_zlabel("z")

    # Plots each metamaterial
    for material in materials:
        # print(material.get_node_positions())
        # print(material.transforms[-1])
        plot_metamaterial(material, subplot=subplot)

    if save:
        plt.savefig(filename)

    # Plays a rotation animation
    if animate:
        for angle in range(0, 360*10):
            subplot.view_init(30, angle)
            plt.draw()
            plt.pause(.002)

    plt.close()


def interpolate(model, material1, material2, interps, path, validate=False):
    """
    Linearly interpolates between the two given materials.

    model: MetamaterialAutoencoder
        The model to use to interpolate.

    material1: Metamaterial
        The base material for interpolation.

    material2: Metamaterial
        The material to be interpolated into.

    interps: int
        The number of interpolations to compute. Includes the starting 
        and ending metamaterial.

    path: str
        The path at which the intermediate metamaterials will be placed.

    validate: bool
        Whether to validate the interpolated metamaterials and remove
        any invalid edges/faces.
    """

    # Computes the latent representation of the two metamaterials
    m1_latent = model.encoder(material1.flatten_rep().reshape((1, NODE_POS_SIZE+EDGE_ADJ_SIZE+FACE_ADJ_SIZE)))
    m2_latent = model.encoder(material2.flatten_rep().reshape((1, NODE_POS_SIZE+EDGE_ADJ_SIZE+FACE_ADJ_SIZE)))

    # Runs through each interpolation
    for ind, alpha in enumerate([x/interps for x in range(interps+1)]):

        # Decodes the interpolated latent representation
        decoding = model.decoder(m1_latent*(1-alpha) + m2_latent*alpha)
        material = Metamaterial.from_tensor(decoding)

        # Validates the decoded representation
        if validate:
            material.remove_invalid_faces() # Removes faces without all edges in the rep
            material.remove_invalid_edges() # Removes edges intersecting with faces
            material.remove_invalid_faces() # Removes faces without all edges in the rep after edge removal

        plot_metamaterial(f"{path}/metamaterial{ind}.png", material, animate=False)
