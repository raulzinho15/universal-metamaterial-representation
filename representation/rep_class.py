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
            in the edge_adj_index() function.
        
        face_adj: ndarray
            The third entry in this tuple is a 1d numpy array face adjacency
            array, where a 1 at the corresponding adjacency tensor's (i,j,k) index
            means that nodes i, j, and k are connected by a triangular face. All other
            entries are 0. The logic for extracting the face adjacencies can be found
            in the face_adj_index() function.
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

        # Stores already-computed node positions on the cube for computation speed-up
        self.cube_pos = {}


    def get_node_position(self, node):
        """
        Computes the 3D position of the given node.

        node: int
            The ID of the node whose position will be returned.

        Returns: ndarray
            The 3D position of the given node.
        """

        # Gets the position of the center node
        if node == NUM_NODES-1:
            x,y,z = (0.5,0.5,0.5)

        elif node in self.cube_pos:
            return self.cube_pos[node]

        # Computes the position of a non-center node
        else:
            theta, phi = self.node_pos[node*2:(node+1)*2]
            theta, phi = theta*np.pi, phi*2*np.pi
            # biases = [0.8,0.5,0.8,0.8]
            x,y,z = project_onto_cube(*spherical_to_euclidian(theta, phi))#, grid_lines=3, bias_cutoff=biases[node])

        # Returns the transformed position
        self.cube_pos[node] = np.array([
            self.translate_x + (1-x if self.mirror_x else x),
            self.translate_y + (1-y if self.mirror_y else y),
            self.translate_z + (1-z if self.mirror_z else z)
        ])
        return self.cube_pos[node]


    def get_node_positions(self):
        """
        Computes the positions of all nodes in the metamaterial representation.

        Returns: ndarray
            The positions of all nodes, ordered as face nodes, edge nodes, vertex
            nodes, and the center node.
        """

        return np.array([list(self.get_node_position(node)) for node in range(NUM_NODES)])
    

    def angle_score(self, node):
        """
        Computes the angle score of the given node.

        node: int
            The node whose angle score will be calculated.

        Returns: float
            The angle score.
        """

        return self.node_pos[2*node] * self.node_pos[2*node+1]
    

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


    def has_edge(self, node1, node2):
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
        return np.array([[self.has_edge(n1, n2)
                        for n1 in range(NUM_NODES)]
                            for n2 in range(NUM_NODES)]).astype(float)


    def has_face(self, node1, node2, node3):
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
    

    def has_some_face(self, node1, node2):
        """
        Checks whether there exists some face between the two nodes.

        node1: int
            The node ID of the first node.

        node2: int
            The node ID of the second node.
        """

        # Checks for a face
        for node3 in range(NUM_NODES):
            if self.has_face(node1, node2, node3):
                return True
            
        # No faces were found
        return False


    def get_face_adj_tensor(self):
        """
        Converts the given face adjacencies into a face adjacency tensor.

        Returns: ndarray
            A face adjacency tensor, where the position (i,j,k) is 1 if and
            only if there is a face connecting the i-th, j-th, and k-th
            nodes. All other entries are 0.
        """

        return np.array([[[self.has_face(n1, n2, n3)
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


    def remove_edges_from_overlaps(self):
        """
        Removes edges from the representation that overlap with a face.
        Mutates the material.
        """

        # Goes through each face
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):

                    # Removes the edge-face overlap
                    if self.has_face(n1, n2, n3):
                        self.edge_adj[edge_adj_index(n1,n2)] = 0
                        self.edge_adj[edge_adj_index(n1,n3)] = 0
                        self.edge_adj[edge_adj_index(n2,n3)] = 0


    def remove_faces_from_overlaps(self):
        """
        Removes faces from the representation that overlap with an edge.
        Mutates the material.
        """

        # Goes through each face
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):

                    # Removes the edge-face overlap
                    if self.has_edge(n1, n2) or self.has_edge(n1, n3) or self.has_edge(n2, n3):
                        self.face_adj[face_adj_index(n1, n2, n3)] = 0


    def nodes_connected_to(self, node):
        """
        Finds the nodes that are connected to the given node by edges/faces.

        node: int
            The node ID of the node whose connected nodes will be found.

        Returns: list of ints
            The list of all nodes connected to the given node, including
            itself.
        """

        # Stores the values for BFS
        queue = [node]
        seen = set([node])

        # Runs BFS while there are still more nodes to be visited
        while queue:
            current = queue.pop(0)

            # Runs through all adjacent nodes
            for n in range(NUM_NODES):

                # Skips seen nodes
                if n in seen:
                    continue

                # Queues the node if adjacent
                if self.has_edge(current, n) or self.has_some_face(current, n):
                    queue.append(n)
                    seen.add(n)
                    continue

        return list(seen)


    def remove_disconnections(self):
        """
        Removes the disconnections from the metamaterial, keeping a
        connected component with the largest number of nodes. Mutates the
        metamaterial.
        """
        
        # Stores the nodes that have yet to be checked
        nodes_left = [i for i in range(NUM_NODES)]

        # Runs through each connected component
        sizes = {}
        while nodes_left:

            # Finds the connected nodes
            current = nodes_left[0]
            connected_nodes = self.nodes_connected_to(current)

            # Removes all the connected nodes from the nodes left to check
            for node in connected_nodes:
                nodes_left.remove(node)

            # Stores the number of nodes in the connected component
            sizes[current] = len(connected_nodes)

        # Finds all nodes in a largest connected component
        max_component = max(sizes, key=lambda x:sizes[x])
        nodes = set(self.nodes_connected_to(max_component))

        # Computes the nodes whose edges must be removed
        nodes_to_remove = [i for i in range(NUM_NODES) if i not in nodes]

        # Removes the edges from each of these nodes
        for n1 in nodes_to_remove:
            for n2 in range(NUM_NODES):

                # Skips self
                if n1 == n2:
                    continue

                # Removes edge
                self.edge_adj[edge_adj_index(n1, n2)] = 0


    def remove_acycles(self):
        """
        Removes the non-connected components of the material. Assumes
        all nodes are reachable from each other.

        NOT WORKING.
        """

        # Stores the cyclic nodes
        cyclic = set()

        # Defines a DFS algorithm for finding the cyclic nodes
        def dfs(node, visited: set, path: tuple):

            # Stores the node as it pertains to the algorithm
            visited.add(node)
            path = path + (node,)

            # Runs through every adjacent node
            for neighbor in range(NUM_NODES):

                # Skips non-neighbor
                if not (self.has_edge(node, neighbor) or self.has_some_face(node, neighbor)):
                    continue

                # Marks as cyclic when cycle is found
                if neighbor in path and neighbor not in visited:
                    for n in path[path.index(neighbor):]:
                        cyclic.add(n)

                # Searches the neighbor if not already searched
                if neighbor not in visited:
                    dfs(neighbor, visited, path)

            # For backtracking
            path = path[:-1]

        # Computes a node that is connected
        start_node = None
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):

                # Checks for an edge connection
                if self.has_edge(n1, n2):
                    start_node = n1
                    break
                        
                # Checks for a face connection
                for n3 in range(n2+1, NUM_NODES):
                    if self.has_face(n1, n2, n3):
                        start_node = n1
                        break

                # Exits if node was found
                if start_node is not None:
                    break

            # Exits if node was found
            if start_node is not None:
                break

        # Checks for cycles
        if start_node is None:
            return
        dfs(start_node, set(), tuple())

        print("Cycles:", cyclic)

        # Disconnects any separate cycles
        for node in self.nodes_connected_to(start_node):

            # Removes if acyclic
            if node not in cyclic:
                for n2 in range(NUM_NODES):
                    
                    # Skips self
                    if node == n2:
                        continue

                    self.edge_adj[edge_adj_index(node, n2)] = 0


    def reorder_nodes(self, node_order):
        """
        Reorders the nodes in the metamaterial. Does not mutate this metamaterial.

        node_order: list or tuple
            The new ordering of the nodes, where for each node_order[i] = j,
            the j-th current node of the metamaterial will become the i-th
            node of the reordered metamaterial.

        Returns: Metamaterial
            The metamaterial with the reordered nodes.
        """
        
        # Stores the reordered node positions
        reordered_node_pos = np.array([self.node_pos[2*node_order[i]:2*node_order[i]+2] for i in range(self.node_pos.shape[0]//2)]).reshape(self.node_pos.shape[0])

        # Stores the reordered edge adjacencies
        reordered_edge_adj = np.zeros(EDGE_ADJ_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                reordered_edge_adj[edge_adj_index(n1, n2)] = self.edge_adj[edge_adj_index(node_order[n1], node_order[n2])]

        # Stores the reordered face adjacencies
        reordered_face_adj = np.zeros(FACE_ADJ_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):
                    reordered_face_adj[face_adj_index(n1, n2, n3)] = self.face_adj[face_adj_index(node_order[n1], node_order[n2], node_order[n3])]

        return Metamaterial(reordered_node_pos, reordered_edge_adj, reordered_face_adj)


    def sort_rep(self):
        """
        Sorts the nodes in increasing order by the product of the two rep angles.

        Returns: Metamaterial
            The sorted metamaterial representation.
        """

        # Stores the sorted node IDs
        sorted_node_indices = sorted(
            [i for i in range(self.node_pos.shape[0]//2)],
                key=lambda node: self.angle_score(node)
        )
        sorted_node_indices.append(NUM_NODES-1) # Includes the center node as last node

        return self.reorder_nodes(sorted_node_indices)


    def best_node_match(self, mat2, nodes1, nodes2):
        """
        Computes the best matching between the first nodes of the given materials.

        mat2: Metamaterial
            The second metamaterial for the best node matching. Assumed to have at
            least as many nodes to match as this metamaterial.

        nodes1: int
            The number of nodes from mat1 (from index 0) that will be compared
            in the matching. Must be <= NUM_NODES and <= nodes2.

        nodes2: int
            The number of nodes from mat2 (from index 0) that will be compared
            in the matching. Must be <= NUM_NODES and >= nodes1.

        Returns: list
            The best matching of each selected node in this metamaterial to nodes
            in mat2. For each output[i] = j, it means mat1's node i should map to
            mat2's node j. Thus, if this list is used to reorder a metamaterial,
            it should be used on mat2's reorder_nodes() function, not mat1's.
        """

        # Stores arrays for the function
        pair = [-1] * nodes1 # The node mapping to it

        # Computes each this node's preferred matches
        score = lambda n1: (lambda n2: np.abs(self.angle_score(n1) - mat2.angle_score(n2)))
        favorites = [sorted([j for j in range(nodes2)], key=score(i)) for i in range(nodes1)]

        # Runs through a matching algorithm
        while -1 in pair:
            for node in range(nodes1):

                # Only tries to make a match if this node does not already have one
                if pair[node] == -1:
                    for favorite in favorites[node]:

                        # Checks if this node can replace the old one
                        if favorite in pair:

                            # Checks if this is worse
                            fav_ind = pair.index(favorite)
                            if score(node)(favorite) >= score(fav_ind)(favorite):
                                continue

                            # Removes the old node
                            pair[fav_ind] = -1

                        # Stores this as the new pair
                        pair[node] = favorite
                        break

        # Reverses the pairings to ensure mat1 maps to mat2 nodes
        # fixed_pair = [0] * nodes1
        # for node in range(nodes1):
        #     fixed_pair[pair[node]] = node
        return pair


    def flatten_rep(self, pad_dim=False):
        """
        Computes the flattened array representation of the metamaterial.

        pad_dim: bool
            Whether an extra dimension will be added for padding to act as a batch of 1.
        """
        concatenation = np.concatenate((self.node_pos, self.edge_adj, self.face_adj))
        if pad_dim:
            concatenation = concatenation.reshape((1,concatenation.shape[0]))
        return torch.from_numpy(concatenation).type(torch.float32)


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

        numpy_rep = rep_tensor.detach().numpy().reshape(NODE_POS_SIZE+EDGE_ADJ_SIZE+FACE_ADJ_SIZE)
        numpy_rep[NODE_POS_SIZE:] = (numpy_rep[NODE_POS_SIZE:] > 0.5).astype(float)
        return Metamaterial(
            numpy_rep[:NODE_POS_SIZE],
            numpy_rep[NODE_POS_SIZE:NODE_POS_SIZE+EDGE_ADJ_SIZE],
            numpy_rep[NODE_POS_SIZE+EDGE_ADJ_SIZE:]
        )
