import numpy as np
import torch
from representation.utils import triangle_line_intersection
from representation.rep_utils import *

class Metamaterial:

    def __init__(self, node_pos, edge_adj, face_adj):
        """
        Initializes a metamaterial representation with the given
        representation arrays.

        node_pos: ndarray
            A 1d numpy array where every row pair or single value is the normalized
            position of its corresponding node on its corresponding unit cube
            face/edge. The nodes appear in the following sequential order:
        
            The first (NODES_PER_FACE * CUBE_FACES) 2D pairs correspond to the nodes
            at the cube's faces. There are the same number of nodes on each cube face.
            The order of cube faces in the array is z=0, z=1, y=0, y=1, x=0, x=1,
            where the coordinates in the node position array appear in the order
            (x,y,z), and exclude whichever dimension whose plane on which the node
            lies (hence 2D). Co-face nodes are next to each other in the array.

            The next (NODES_PER_EDGE * CUBE_EDGES) values correspond to the nodes at
            the cube's edges. There are the same number of nodes on each edge. The
            order of edges in the array is (x=0, y=0), (x=0, y=1), (x=1, y=0),
            (x=1, y=1), (x=0, z=0), (x=0, z=1), (x=1, z=0), (x=1, z=1), (y=0, z=0),
            (y=0, z=1), (y=1, z=0), (y=1, z=1), where the coordinate stored in this
            array is in the non-constant axis for that node. Co-edge nodes are next
            to each other in the array.

            The ordering of nodes in this array is the ordering that is logically
            used when using the edge/face adjacency representations. The logic for
            extracting nodes' positions can be found in the get_node_x/y/z() functions.
        
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


    def __get_node_x(self, node):
        """
        Computes what the untransformed x coordinate of the node with the given ID is
        based on the given node positions.

        node: int
            The ID of the node.

        Returns: float
            The x coordinate of the node, untransformed.
        """

        # Cube center node
        if node == NUM_NODES-1:
            return 0.5
        
        # Cube face node
        if node < NODES_PER_FACE * CUBE_FACES:
            if node // (2*NODES_PER_FACE) != 2: # Not in a face parallel to the yz plane
                return self.node_pos[node*2]
            return (node//NODES_PER_FACE) % 2 # In a face parallel to the yz plane
        
        # Cube edge node
        node -= NODES_PER_FACE * CUBE_FACES # Offsets the node into the edge section
        if node < NODES_PER_EDGE * CUBE_EDGES:
            if node // (4*NODES_PER_EDGE) == 2: # In an edge that runs along the x axis
                return self.node_pos[NODES_PER_FACE * CUBE_FACES * 2 + node]
            return (node // (2*NODES_PER_EDGE)) % 2 # In an edge that does not run along the x axis
        
        # Cube vertex node
        node -= NODES_PER_EDGE * CUBE_EDGES # Offsets the node into the vertex section
        return node//4


    def __get_node_y(self, node):
        """
        Computes what the y untransformed coordinate of the node with the given ID is
        based on the given node positions, untransformed.

        node: int
            The ID of the node.

        Returns: float
            The y coordinate of the node.
        """

        # Cube center node
        if node == NUM_NODES-1:
            return 0.5
        
        # Cube face node
        if node < NODES_PER_FACE * CUBE_FACES:
            if node // (2*NODES_PER_FACE) != 1: # Not in a face parallel to the xz plane
                return self.node_pos[node*2 + (1 if node // (2*NODES_PER_FACE) == 0 else 0)]
            return (node//NODES_PER_FACE) % 2 # In a face parallel to the xz plane
        
        # Cube edge node
        node -= NODES_PER_FACE * CUBE_FACES # Offsets the node into the edge section
        if node < NODES_PER_EDGE * CUBE_EDGES:
            if node // (4*NODES_PER_EDGE) == 1: # In an edge that runs along the y axis
                return self.node_pos[NODES_PER_FACE * CUBE_FACES * 2 + node]
            return (node // ((2 if node // (4*NODES_PER_EDGE) == 2 else 1) * NODES_PER_EDGE)) % 2 # In an edge that does not run along the y axis
        
        # Cube vertex node
        node -= NODES_PER_EDGE * CUBE_EDGES # Offsets the node into the vertex section
        return (node//2) % 2


    def __get_node_z(self, node):
        """
        Computes what the z untransformed coordinate of the node with the given ID is
        based on the given node positions.

        node: int
            The ID of the node.

        Returns: float
            The z coordinate of the node.
        """

        # Cube center node
        if node == NUM_NODES-1:
            return 0.5
        
        # Cube face node
        if node < NODES_PER_FACE * CUBE_FACES:
            if node // (2*NODES_PER_FACE) != 0: # Not in a face parallel to the xy plane
                return self.node_pos[node*2+1]
            return (node//NODES_PER_FACE) % 2 # In a face parallel to the xy plane
        
        # Cube edge node
        node -= NODES_PER_FACE * CUBE_FACES # Offsets the node into the edge section
        if node < NODES_PER_EDGE * CUBE_EDGES:
            if node // (4*NODES_PER_EDGE) == 0: # In an edge that runs along the z axis
                return self.node_pos[NODES_PER_FACE * CUBE_FACES * 2 + node]
            return (node // NODES_PER_EDGE) % 2 # In an edge that does not run along the z axis
        
        # Cube vertex node
        node -= NODES_PER_EDGE * CUBE_EDGES # Offsets the node into the vertex section
        return node%2
    

    def get_node_position(self, node):
        """
        Computes the 3D position of the given node.

        node: int
            The ID of the node whose position will be returned.

        Returns: 3-tuple of floats
            The 3D position of the given node.
        """
        x, y, z = self.__get_node_x(node), self.__get_node_y(node), self.__get_node_z(node)
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

        return np.array([[[self.have_face(n1, n2, n3, self.face_adj)
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


    def flatten_rep(self):
        """
        Computes the flattened array representation of the metamaterial.
        """
        concatenation = np.concatenate((self.node_pos, self.edge_adj, self.face_adj))
        return torch.from_numpy(concatenation).type(torch.float32)


    def copy(self):
        """
        Creates a copy of the Metamaterial.

        Returns: Metamaterial
            A copy of this metamaterial
        """

        # Copies the representation arrays
        material = Metamaterial(np.copy(self.node_pos), np.copy(self.edge_adj), np.copy(self.face_adj))

        # Copies the mirror transforms
        material.mirror_x = self.mirror_x
        material.mirror_y = self.mirror_y
        material.mirror_z = self.mirror_z
        for transform in self.transforms:
            material.apply_transform(transform)
        return material
    
    def from_tensor(rep_tensor):
        """
        Creates a Metamaterial from the given PyTorch tensor.

        rep_tensor: Tensor
            A tensor with the representation arrays concatenated together.
        """

        numpy_rep = rep_tensor[0,:].detach().numpy()
        return Metamaterial(
            numpy_rep[:NODE_POS_SIZE],
            numpy_rep[NODE_POS_SIZE:NODE_POS_SIZE+EDGE_ADJ_SIZE],
            np.zeros(FACE_ADJ_SIZE))
            # numpy_rep[NODE_POS_SIZE+EDGE_ADJ_SIZE:])
