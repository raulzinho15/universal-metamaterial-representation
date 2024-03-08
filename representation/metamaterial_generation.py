import numpy as np
import matplotlib.pyplot as plt
from rep_class import *
from rep_utils import *
from random import random


def random_metamaterial(edge_prob=0.5, face_prob=0.5, with_faces=True):
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
    # metamaterial.remove_invalid_faces() # Removes faces without all edges in the rep
    # metamaterial.remove_invalid_edges() # Removes edges intersecting with faces
    # metamaterial.remove_invalid_faces() # Removes faces without all edges in the rep after edge removal

    return metamaterial

# Computes consistent colors for each edge
colors = {}
for n1 in range(NUM_NODES):
    for n2 in range(n1+1, NUM_NODES):
        colors[(n1, n2)] = tuple(random() for _ in range(3))

def plot_metamaterial(filename, metamaterial, animate=True):
    """
    Plots the metamaterial with the given representation at the given filename.

    filename: str
        The name of the file at which the plot image will be saved.

    metamaterial: Metamaterial
        The metamaterial to plot.
    """
    
    # Sets up the 3d plot environment
    fig = plt.figure()
    plot3d = fig.add_subplot(projection="3d")
    plot3d.set_xlabel("x")
    plot3d.set_ylabel("y")
    plot3d.set_zlabel("z")

    # Plots each edge
    for n1 in range(NUM_NODES):
        for n2 in range(n1+1, NUM_NODES):

            # Skips unconnected nodes
            if not metamaterial.have_edge(n1, n2):
                continue

            # Computes the edge coordinates
            x1, y1, z1 = metamaterial.get_node_position(n1)
            x2, y2, z2 = metamaterial.get_node_position(n2)

            plot3d.plot([x1, x2], [y1, y2], zs=[z1, z2], linewidth=5, color=colors[(n1, n2)])

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

                plot3d.plot_trisurf([x1, x2, x3], [y1, y2, y3], [z1, z2, z3], alpha=0.4)

    plt.savefig(filename)

    # Plays a rotation animation
    if animate:
        for angle in range(0, 360*10):
            plot3d.view_init(30, angle)
            plt.draw()
            plt.pause(.002)

    plt.close()
