import numpy as np
import torch
from rotation_rep.representation.utils import triangle_line_intersection
from rotation_rep.representation.rep_utils import *

class Metamaterial:

    def __init__(self, node_pos, edge_adj, face_adj):
        """
        Initializes a metamaterial representation with the given
        representation arrays.

        node_pos: ndarray
            A 1d numpy array where every row pair is a 2D spherical angular
            coordinates of the form (theta, phi).

            The ordering of nodes in this array is the ordering that is logically
            used when using the edge/face adjacency representations. The logic for
            extracting nodes' positions can be found in get_node_positions().
        
        edge_adj: ndarray
            The second entry in this tuple is a 1d numpy array edge adjacency
            array, where a 1 at the corresponding adjacency matrix's i-th row and
            j-th column means that nodes i and j are connected by an edge. All other
            entries are 0. The logic for extracting the edge adjacencies can be found
            in the have_edge() function.
        
        face_adj: ndarray
            The third entry in this tuple is a 1d numpy array face adjacency
            array, where a 1 at the corresponding adjacency tensor's (i,j,k) index
            means that nodes i, j, and k are connected by a triangular face. All other
            entries are 0. The logic for extracting the face adjacencies can be found
            in the have_face() function.
        """

        # Stores the rep arrays
        self.node_pos = np.copy(node_pos)
        self.edge_adj = np.copy(edge_adj)
        self.face_adj = np.copy(face_adj)

        # Stores mirror transforms
        self.mirror_x = False
        self.mirror_y = False
        self.mirror_z = False

        # Stores translation transforms
        self.translate_x = 0
        self.translate_y = 0
        self.translate_z = 0


    def get_node_position(self, node):
        """
        Computes the 3D position of the given node.

        node: int
            The ID of the node whose position will be returned.

        Returns: 3-tuple of floats
            The 3D position of the given node.
        """

        # Gets the position of the center node
        if node == NUM_NODES-1:
            x,y,z = (0.5,0.5,0.5)

        # Computes the position of a non-center node
        else:
            theta, phi = self.node_pos[node*2:(node+1)*2]
            theta, phi = theta*np.pi, phi*2*np.pi
            pos = spherical_to_euclidian(theta, phi)
            pos /= np.max(np.abs(pos))
            x, y, z = (pos + np.ones(3)) / 2
            
            # x = 1 if x > (1-1e-4) else 0 if x < 1e-4 else x
            # y = 1 if y > (1-1e-4) else 0 if y < 1e-4 else y
            # z = 1 if z > (1-1e-4) else 0 if z < 1e-4 else z

        # Returns the transformed position
        return (
            self.translate_x + (1-x if self.mirror_x else x),
            self.translate_y + (1-y if self.mirror_y else y),
            self.translate_z + (1-z if self.mirror_z else z)
        )


    def get_node_positions(self):
        """
        Computes the positions of all nodes in the metamaterial representation.

        Returns: ndarray
            The positions of all nodes, ordered as face nodes, edge nodes, vertex
            nodes, and the center node.
        """

        return np.array([list(self.get_node_position(node)) for node in range(NUM_NODES)])
    

    def mirror(self, x=False, y=False, z=False):
        """
        Mirrors each node across a chosen plane going through the center of the
        metamaterial. Does not mutate this metamaterial.

        x: bool
            Whether the x coordinates will be mirrored.

        y: bool
            Whether the y coordinates will be mirrored.

        z: bool
            Whether the z coordinates will be mirrored.

        Returns: Metamaterial
            A copy of this metamaterial with the mirrored nodes.
        """

        # Creates copy for mirroring
        material = self.copy()

        # Mirrors the coordinates
        material.mirror_x = not material.mirror_x if x else material.mirror_x
        material.mirror_y = not material.mirror_y if y else material.mirror_y
        material.mirror_z = not material.mirror_z if z else material.mirror_z

        return material
    

    def translate(self, dx=0,dy=0,dz=0):
        """
        Translates each node by the given displacement. Does not mutate
        this metamaterial.

        dx: int
            The amount by which the x coordinate will be translated.

        dy: int
            The amount by which the y coordinate will be translated.

        dz: int
            The amount by which the z coordinate will be translated.

        Returns: Metamaterial
            A copy of this metamaterial with the translated nodes.
        """

        # Creates copy for translating
        material = self.copy()

        # Translates the coordinates
        material.translate_x += dx
        material.translate_y += dy
        material.translate_z += dz

        return material


    def have_edge(self, node1, node2):
        """
        Checks whether the two given nodes have an edge between them
        based on the edge adjacencies.

        node1: int
            The ID of the first node.

        node2: int
            The ID of the second node.

        Returns: bool
            Whether there is an edge between the two given nodes.
        """

        # Trivially excludes an edge from identical nodes
        if node1 == node2:
            return False

        return self.edge_adj[edge_adj_index(node1, node2)] == 1


    def get_edge_adj_matrix(self):
        """
        Converts the edge adjacencies into an edge adjacency matrix.

        Returns: ndarray
            An edge adjacency matrix, where the position (i,j) is 1 if and
            only if there is an edge connecting the i-th and j-th nodes.
            All other entries are 0.
        """
        return np.array([[self.have_edge(n1, n2)
                        for n1 in range(NUM_NODES)]
                            for n2 in range(NUM_NODES)]).astype(float)


    def have_face(self, node1, node2, node3):
        """
        Checks whether the three given nodes have a face between them
        based on the face adjacencies.

        node1: int
            The ID of the first node.

        node2: int
            The ID of the second node.

        node3: int
            The ID of the third node.

        Returns: bool
            Whether there is a face between the three given nodes.
        """

        # Trivially excludes a face from identical nodes
        if node1 == node2 or node2 == node3 or node1 == node3:
            return False

        return self.face_adj[face_adj_index(node1, node2, node3)] == 1

    def get_face_adj_tensor(self):
        """
        Converts the given face adjacencies into a face adjacency tensor.

        Returns: ndarray
            A face adjacency tensor, where the position (i,j,k) is 1 if and
            only if there is a face connecting the i-th, j-th, and k-th
            nodes. All other entries are 0.
        """

        return np.array([[[self.have_face(n1, n2, n3)
                        for n1 in range(NUM_NODES)]
                            for n2 in range(NUM_NODES)]
                                for n3 in range(NUM_NODES)]).astype(float)


    def remove_invalid_faces(self):
        """
        Removes all the faces described in the face adjacency array whose
        three edges are not encoded in the edge adjacency.
        """

        # Checks if every face validly has its corresponding 3 edges
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):

                    # Skips non-face edge triplets
                    index = face_adj_index(n1, n2, n3)
                    if not self.face_adj[index]:
                        continue

                    # Removes the invalid face (i.e., a face with insufficient edges)
                    if not (self.edge_adj[edge_adj_index(n1, n2)]
                        and self.edge_adj[edge_adj_index(n1, n3)]
                        and self.edge_adj[edge_adj_index(n2, n3)]):
                        self.face_adj[index] = 0


    def remove_invalid_edges(self):
        """
        Removes all the edges described in the edge adjacency array which
        intersect with faces described in the face adjacency array.
        """

        # Runs through each possible face
        for nf1 in range(NUM_NODES):
            for nf2 in range(nf1+1, NUM_NODES):
                for nf3 in range(nf2+1, NUM_NODES):

                    # Skips non-faces
                    if not self.face_adj[face_adj_index(nf1, nf2, nf3)]:
                        continue

                    # Runs through each possible edge
                    face_nodes = (nf1, nf2, nf3)
                    for ne1 in range(NUM_NODES):

                        # Skips node on the face
                        if ne1 in face_nodes:
                            continue

                        for ne2 in range(ne1+1, NUM_NODES):

                            # Skips node on the face
                            if ne2 in face_nodes:
                                continue

                            # Skips non-edges
                            index = edge_adj_index(ne1, ne2)
                            if not self.edge_adj[index]:
                                continue

                            # Checks for intersection
                            positions = [np.array(list(self.get_node_position(n)))
                                            for n in (nf1, nf2, nf3, ne1, ne2)]
                            if triangle_line_intersection(*positions):
                                self.edge_adj[index] = 0


    def sort_rep(self):
        """
        Sorts the nodes by the product of the two rep angles.

        Returns: Metamaterial
            The sorted metamaterial representation.
        """

        # Stores the sorted node IDs
        sorted_node_indices = sorted(
            [i for i in range(self.node_pos.shape[0]//2)],
                key=lambda x: self.node_pos[2*x]*self.node_pos[2*x+1]
        )
        sorted_node_indices.append(NUM_NODES-1) # Includes the center node as last node

        # Stores the sorted node positions
        sorted_node_pos = np.array(
            [self.node_pos[2*sorted_node_indices[i]:2*sorted_node_indices[i]+2] for i in range(self.node_pos.shape[0]//2)]
        ).reshape(self.node_pos.shape[0])

        # Stores the sorted edge adjacencies
        sorted_edge_adj = np.zeros(EDGE_ADJ_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                sorted_edge_adj[edge_adj_index(n1, n2)] = self.edge_adj[edge_adj_index(sorted_node_indices[n1], sorted_node_indices[n2])]

        # Stores the sorted face adjacencies
        sorted_face_adj = np.zeros(FACE_ADJ_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):
                    sorted_face_adj[face_adj_index(n1, n2, n3)] = self.face_adj[face_adj_index(sorted_node_indices[n1], sorted_node_indices[n2], sorted_node_indices[n3])]

        return Metamaterial(sorted_node_pos, sorted_edge_adj, sorted_face_adj)


    def flatten_rep(self):
        """
        Computes the flattened array representation of the metamaterial.
        """
        concatenation = np.concatenate((self.node_pos, self.edge_adj, self.face_adj))
        return torch.from_numpy(concatenation).type(torch.float32)#[:NODE_POS_SIZE+EDGE_ADJ_SIZE]


    def copy(self):
        """
        Creates a copy of the Metamaterial.

        Returns: Metamaterial
            A copy of this metamaterial
        """

        # Copies the representation arrays
        material = Metamaterial(self.node_pos, self.edge_adj, self.face_adj)

        # Copies the mirror transforms
        material.mirror_x = self.mirror_x
        material.mirror_y = self.mirror_y
        material.mirror_z = self.mirror_z

        # Copies the translate transforms
        material.translate_x = self.translate_x
        material.translate_y = self.translate_y
        material.translate_z = self.translate_z

        return material
    
    def from_tensor(rep_tensor):
        """
        Creates a Metamaterial from the given PyTorch tensor.

        rep_tensor: Tensor
            A tensor with the representation arrays concatenated together.
        """

        numpy_rep = rep_tensor.detach().numpy()
        numpy_rep[NODE_POS_SIZE:] = (numpy_rep[NODE_POS_SIZE:] > 0.5).astype(float)
        return Metamaterial(
            numpy_rep[:NODE_POS_SIZE],
            numpy_rep[NODE_POS_SIZE:NODE_POS_SIZE+EDGE_ADJ_SIZE],
            # np.zeros(FACE_ADJ_SIZE))
            numpy_rep[NODE_POS_SIZE+EDGE_ADJ_SIZE:])
