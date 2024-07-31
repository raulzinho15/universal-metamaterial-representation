import numpy as np
import torch
from random import randrange
from representation.rep_utils import *
from representation.utils import *
from typing import Self

class Metamaterial:

    def __init__(self, node_pos: np.ndarray, edge_adj: np.ndarray, edge_params: np.ndarray, face_adj: np.ndarray, face_params: np.ndarray, thickness=1.0):
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
            A 1d numpy array edge adjacency array, where a 1 at the corresponding
            adjacency matrix's i-th row and j-th column means that nodes i and j are
            connected by an edge. All other entries are 0. The logic for extracting
            the edge adjacencies can be found in the edge_adj_index() function.

        edge_params: ndarray
            A 1d numpy array with (scaled) coefficients for Bezier curves, where
            groups of EDGE_BEZIER_PARAMS coefficients will be associated with the
            edge in the corresponding position. Specifically, the non-node
            coefficients of the Bezier curve will be set to those values.
        
        face_adj: ndarray
            A 1d numpy array face adjacency array, where a 1 at the corresponding
            adjacency tensor's (i,j,k) index means that nodes i, j, and k are connected
            by a triangular face. All other entries are 0. The logic for extracting the
            face adjacencies can be found in the face_adj_index() function.

        face_params: ndarray
            A 1d numpy array with (scaled) coefficients for Bezier triangle, where
            groups of FACE_BEZIER_PARAMS coefficients will be associated with the
            edge in the corresponding position. Specifically, the non-edge
            coefficients of the Bezier curve will be set to those values.

        thickness: float, optional
            The proportion of `THICKNESS` to be used. Must be in `[0,1]`.
            The default is 1.
        """

        # Stores the rep arrays
        self.node_pos = np.copy(node_pos)
        self.edge_adj = np.copy(edge_adj)
        self.edge_params = np.copy(edge_params)
        self.face_adj = np.copy(face_adj)
        self.face_params = np.copy(face_params)

        # Stores the thickness
        self.thickness = thickness

        # Stores mirror transforms
        self.mirrors = np.zeros((1,3))

        # Stores translation transforms
        self.translations = np.zeros((1,3))

        # Stores the displayed planes
        # The order is as follows: x=0, y=0, z=0, x=1, y=1, z=1
        self.displayed_planes = np.array([True] * 6)

        # Stores already-computed node positions on the cube for computation speed-up
        self.cube_pos = {}

        # Stores the mapping from the original node ordering to the
        # new space of new ordering, if a node reordering was applied
        # Meaning: original node node_ordering[i] is now mapped to current node i
        # (same definition as in the reorder_nodes() function)
        self.node_ordering = [i for i in range(NUM_NODES)] # Default: each node maps to itself


    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms the given point(s) with mirroring and translations
        according to the metamaterial's transforms.

        points: np.ndarray
            The point(s) to be transformed. If multiple points,
            then should be a flattened array.

        Returns: np.ndarray
            The transformed point(s) as a flattened array.
        """

        # Applies the transformations
        points = points.reshape(points.shape[0]//3, 3)
        return (self.translations + (1-points)*self.mirrors + points*(1-self.mirrors)).flatten() * SCALE


    def get_node_position(self, node: int, transform=True) -> np.ndarray:
        """
        Computes the 3D position of the given node.

        node: int
            The ID of the node whose position will be returned.

        transform: `bool`, optional
            Whether the node's position will have transformations applied.

        Returns: np.ndarray
            A 1d numpy array containing the 3D position of the given node.
        """

        # Uses a pre-computed node position
        if node in self.cube_pos:
            return self.cube_pos[node]

        # Gets the position of the node
        point = pseudo_spherical_to_euclidean(self.node_pos[node*3 : (node+1)*3][np.newaxis,:]).flatten()

        # Transforms the position
        if transform:
            point = self.transform_points(point)

        # Returns the position
        self.cube_pos[node] = point
        return self.cube_pos[node]


    def get_node_positions(self) -> np.ndarray:
        """
        Computes the positions of all nodes in the metamaterial representation.

        Returns: ndarray
            The positions of all nodes, ordered as according to node ID.
        """

        return np.array([self.get_node_position(node) for node in range(NUM_NODES)])
    

    def angle_score(self, node: int) -> float:
        """
        Computes the angle score of the given node.

        node: int
            The node whose angle score will be calculated.

        Returns: float
            The angle score.
        """

        return self.node_pos[3*node] * self.node_pos[3*node+1]
    

    def get_thickness(self) -> float:
        """
        Computes the thickness of nodes, edges, and faces on this metamaterial.

        Returns: float
            The thickness of this metamaterial.
        """

        return self.thickness * THICKNESS
    

    def set_thickness(self, thickness: float) -> Self:
        """
        Makes a copy of this metamaterial with the given thickness.

        thickness: float
            The thickness to set for the new metamaterial.

        Returns: Metamaterial
            The new metamaterial with the given thickness.
        """

        material = self.copy()
        material.thickness = thickness
        return material
    

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

        # Prepares a numpy array of the mirrorings
        new_mirrors = np.array([[x, y, z]])

        # Mirrors the coordinates
        material.mirrors = np.logical_xor(new_mirrors, material.mirrors)

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

        # Creates a numpy array of the translations
        new_translations = np.array([[dx, dy, dz]])

        # Translates the coordinates
        material.translations += new_translations

        return material
    

    def toggle_plane_display(self, x0=False, x1=False, y0=False, y1=False, z0=False, z1=False):
        """
        Toggles the planes to be displayed when meshing the material.
        Namely, any edges and faces that lie on a non-displayed plane
        will not be included in the material's mesh.

        x0: `bool`
            Whether to toggle the x=0 plane's display.

        x1: `bool`
            Whether to toggle the x=1 plane's display.

        y0: `bool`
            Whether to toggle the y=0 plane's display.

        y1: `bool`
            Whether to toggle the y=1 plane's display.

        z0: `bool`
            Whether to toggle the z=0 plane's display.

        z1: `bool`
            Whether to toggle the z=1 plane's display.
        """

        # Creates copy for toggling the plane displays
        material = self.copy()

        # Stores the plane display toggles
        display_toggles = np.array([x0, y0, z0, x1, y1, z1])

        # Stores the new plane display properties
        material.displayed_planes = np.logical_xor(material.displayed_planes, display_toggles)

        return material


    def point_is_displayed(self, point: np.ndarray, threshold=1e-4) -> bool:
        """
        Checks whether the given point is on a displayed plane.

        point: np.ndarray
            The point to check.

        threshold: `float`, optional
            The maximum distance a point can have from a plane
            to still be considered to be on that plane.
            Must be positive.

        Returns: `bool`
            Whether the given point is on a displayed plane.
        """

        # Checks whether the point is on the planes
        point_is_on_planes = np.concatenate((np.abs(point) < threshold, np.abs(point-1) < threshold))

        # Checks if the node should be displayed
        return not (point_is_on_planes * (1-self.displayed_planes)).any()
    

    def node_is_displayed(self, node: int, threshold=1e-4) -> bool:
        """
        Checks whether the given node is on a displayed plane.

        node: int
            The node ID of the node to check.

        threshold: `float`, optional
            The maximum distance a point can have from a plane
            to still be considered to be on that plane.
            Must be positive.

        Returns: `bool`
            Whether the given node is on a displayed plane.
        """

        # Computes the untransformed node position
        node_pos = self.get_node_position(node)

        # Checks whether the node should be displayed
        return self.point_is_displayed(node_pos-self.translations.flatten(), threshold=threshold)
    

    def edge_is_displayed(self, node1: int, node2: int, threshold=1e-4) -> bool:
        """
        Checks whether the given edge is on a displayed plane.

        node1: int
            The node ID of the first node of the edge to check.

        node2: int
            The node ID of the second node of the edge to check.
            Must be less than `node1`.

        threshold: `float`, optional
            The maximum distance a point can have from a plane
            to still be considered to be on that plane.
            Must be positive.

        Returns: `bool`
            Whether the given edge is on a displayed plane.
        """

        # Prepares the edge point function
        edge_function = self.compute_edge_points(node1, node2)

        # Computes a pseudo-average point along the edge
        avg_point = sum(
            edge_function(randrange(EDGE_SEGMENTS+1)) for _ in range(EDGE_SEGMENTS//4)
        ) / (EDGE_SEGMENTS//4)

        # Checks whether the edge should be displayed
        return self.point_is_displayed(avg_point-self.translations.flatten(), threshold=threshold)
    

    def face_is_displayed(self, node1: int, node2: int, node3: int, threshold=1e-4) -> bool:
        """
        Checks whether the given face is on a displayed plane.

        node1: int
            The node ID of the first node of the face to check.
            Must be less than other two node IDs.

        node2: int
            The node ID of the second node of the face to check.

        node3: int
            The node ID of the third node of the face to check.
            Must be greater than other two node IDs.

        threshold: `float`, optional
            The maximum distance a point can have from a plane
            to still be considered to be on that plane.
            Must be positive.

        Returns: `bool`
            Whether the given face is on a displayed plane.
        """

        # Prepares the face point function
        face_function = self.compute_face_points(node1, node2, node3)

        # Computes a pseudo-average point along the face
        avg_point = np.zeros(3)
        for _ in range(EDGE_SEGMENTS//4):
            s = randrange(EDGE_SEGMENTS+1)
            t = randrange(EDGE_SEGMENTS+1-s)
            avg_point += face_function(s,t)
        avg_point /= EDGE_SEGMENTS//4

        # Checks whether the face should be displayed
        return self.point_is_displayed(avg_point-self.translations.flatten(), threshold=threshold)


    def has_edge(self, node1: int, node2: int) -> bool:
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
    

    def get_edge_params(self, node1: int, node2: int, transform=True) -> np.ndarray:
        """
        Gets the parameters of the edge between the two given nodes.
        Assumes an edge exists between the two nodes.

        node1: int
            The ID of the first node of the edge.

        node2: int
            The ID of the second node of the edge.

        transform: `bool`, optional
            Whether the edge's parameters will have transformations applied.

        Returns: np.ndarray
            The edge parameters of the edge, transformed according
            to the metamaterial's transformations.
        """

        # Retrieves the edge parameters
        edge_index = edge_adj_index(node1, node2) * EDGE_BEZIER_COORDS
        params = self.edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS]

        # Applies the transform
        if transform:
            params = self.transform_points(params)

        return params
    

    def rotate_edge_params(self, node1: int, node2: int, axis: np.ndarray, angle: float):
        """
        Rotates the edge parameters by the given amount. Does mutate
        the metamaterial.

        node1: `int`
            The ID of the first node of the edge.

        node2: `int`
            The ID of the second node of the edge.

        axis: `ndarray`
            The axis along which the rotation will happen.

        angle: `float`
            The angle by which the rotation will happen, according
            to the right-hand rule.
        """

        # Stores the edge parameters
        edge_params = self.get_edge_params(node1, node2).reshape(-1, 3)

        # Rotates the edge params
        rotated_edge_params = rotate_around_axis(edge_params, axis, angle).flatten()

        # Stores the rotated edge parameters
        edge_index = edge_adj_index(node1, node2)
        self.edge_params[edge_index*EDGE_BEZIER_COORDS : (edge_index+1)*EDGE_BEZIER_COORDS] = rotated_edge_params


    def get_edge_adj_matrix(self) -> np.ndarray:
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
    

    def compute_edge_points(self, node1: int, node2: int):
        """
        Computes the coordinates of the points that make up an edge.

        node1: int
            The node ID of the first node of the edge.

        node2: int
            The node ID of the second node of the edge.

        Returns: function or None
            A function giving the coordinate of the edge point for a given t
            along on the Bezier curve. If no edge exists, returns None.
        """

        # Computes the edge's nodes' positions
        node1, node2 = sorted((node1, node2), key=lambda n: self.node_ordering[n])
        node1_pos = self.get_node_position(node1)[np.newaxis,:]
        node2_pos = self.get_node_position(node2)[np.newaxis,:]
        
        # Retrieves the edge parameters
        edge_params = self.get_edge_params(node1, node2).reshape((EDGE_BEZIER_POINTS,3)) + node1_pos

        # Appropriately structures all parameters for the Bezier curve
        bezier_params = np.concatenate((node1_pos, edge_params, node2_pos), axis=0)

        # Creates the function to compute the edge points
        def bezier(t: int) -> np.ndarray:
            return BEZIER_CURVE_COEFFICIENTS[t,:] @ bezier_params
        
        return bezier


    def has_face(self, node1: int, node2: int, node3: int) -> bool:
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
    

    def has_some_face(self, node1: int, node2: int) -> bool:
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
    

    def get_face_params(self, node1: int, node2: int, node3) -> np.ndarray:
        """
        Gets the parameters of the face between the three given nodes.

        node1: int
            The ID of the first node of the face.

        node2: int
            The ID of the second node of the face.

        node3: int
            The ID of the third node of the face.

        Returns: np.ndarray
            The face parameters of the face if it exists, otherwise `None`.
        """

        # Retrieves the face parameters
        face_index = face_adj_index(node1, node2, node3) * FACE_BEZIER_COORDS
        return self.transform_points(self.face_params[face_index : face_index + FACE_BEZIER_COORDS])


    def get_face_adj_tensor(self) -> np.ndarray:
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
    

    def compute_face_points(self, node1: int, node2: int, node3: int):
        """
        Computes the coordinates of the points that make up a face.

        node1: int
            The node ID of the first node of the face.

        node2: int
            The node ID of the second node of the face.

        node3: int
            The node ID of the third node of the face.

        Returns: function or None
            A function giving the coordinate of the face point for a given s,t
            along on the Bezier triangle. If no face exists, returns None.
        """
        
        # Checks for face existence
        if not self.has_face(node1, node2, node3):
            return None

        # Stores the node positions
        node1, node2, node3 = sorted((node1, node2, node3), key=lambda n: self.node_ordering[n])
        node1_pos = self.get_node_position(node1)[np.newaxis, :]
        node2_pos = self.get_node_position(node2)[np.newaxis, :]
        node3_pos = self.get_node_position(node3)[np.newaxis, :]

        # Stores the edge parameters
        edge1_params = self.get_edge_params(node1, node2).reshape((EDGE_BEZIER_POINTS, 3))
        edge2_params = self.get_edge_params(node1, node3).reshape((EDGE_BEZIER_POINTS, 3))
        edge3_params = self.get_edge_params(node2, node3).reshape((EDGE_BEZIER_POINTS, 3))
        
        # Stores the face parameters
        face_params = self.get_face_params(node1, node2, node3).reshape((FACE_BEZIER_POINTS, 3))

        # Appropriately structures all parameters for the Bezier curve
        bezier_params = np.concatenate([
            node1_pos, node2_pos, node3_pos,
            edge1_params, edge2_params, edge3_params,
            face_params
        ], axis=0)

        # Creates the function to compute the face points
        def bezier(s: int, t: int) -> np.ndarray:
            ind = bezier_triangle_index(s,t)
            return BEZIER_TRIANGLE_COEFFICIENTS[ind,:] @ bezier_params
        return bezier


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

        # Disconnects any separate cycles
        for node in self.nodes_connected_to(start_node):

            # Removes if acyclic
            if node not in cyclic:
                for n2 in range(NUM_NODES):
                    
                    # Skips self
                    if node == n2:
                        continue

                    self.edge_adj[edge_adj_index(node, n2)] = 0


    def reorder_nodes(self, node_order: list[int]):
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
        reordered_node_pos = np.concatenate([self.node_pos[3*node_order[i]:3*node_order[i]+3] for i in range(self.node_pos.shape[0]//3)])

        # Stores the reordered edge adjacencies
        reordered_edge_adj = np.zeros(EDGE_ADJ_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                reordered_edge_adj[edge_adj_index(n1, n2)] = self.edge_adj[edge_adj_index(node_order[n1], node_order[n2])]

        # Stores the reordered edge parameters
        reordered_edge_params = np.zeros(EDGE_PARAMS_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                reordered_index = edge_adj_index(n1, n2) * EDGE_BEZIER_COORDS
                old_index = edge_adj_index(node_order[n1], node_order[n2]) * EDGE_BEZIER_COORDS
                reordered_edge_params[reordered_index:reordered_index+EDGE_BEZIER_COORDS] = self.edge_params[old_index:old_index+EDGE_BEZIER_COORDS]

        # Stores the reordered face adjacencies
        reordered_face_adj = np.zeros(FACE_ADJ_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):
                    reordered_face_adj[face_adj_index(n1, n2, n3)] = self.face_adj[face_adj_index(node_order[n1], node_order[n2], node_order[n3])]

        # Stores the reordered face parameters
        reordered_face_params = np.zeros(FACE_PARAMS_SIZE)
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):
                    reordered_index = face_adj_index(n1, n2, n3) * FACE_BEZIER_COORDS
                    old_index = face_adj_index(node_order[n1], node_order[n2], node_order[n3]) * FACE_BEZIER_COORDS
                    reordered_face_params[reordered_index:reordered_index+FACE_BEZIER_COORDS] = self.face_params[old_index:old_index+FACE_BEZIER_COORDS]

        # Stores the original node mapping to the new node space
        new_ordering = [0] * NUM_NODES
        for n in range(NUM_NODES):
            new_ordering[n] = self.node_ordering[node_order[n]]
        self.node_ordering = new_ordering

        return Metamaterial(reordered_node_pos, reordered_edge_adj, reordered_edge_params, reordered_face_adj, reordered_face_params, thickness=self.thickness)


    def sort_rep(self) -> Self:
        """
        Sorts the nodes in increasing order by the product of the two rep angles.

        Returns: Metamaterial
            The sorted metamaterial representation.
        """

        # Stores the sorted node IDs
        sorted_node_indices = sorted(
            [i for i in range(self.node_pos.shape[0]//3)],
                key=lambda node: self.angle_score(node)
        )
        sorted_node_indices.append(NUM_NODES-1) # Includes the center node as last node

        return self.reorder_nodes(sorted_node_indices)


    def best_node_match(self, mat2: Self, nodes1: int, nodes2: int) -> list[int]:
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

        Returns: list[int]
            The best matching of each selected node in this metamaterial to nodes
            in mat2. For each output[i] = j, it means this material's node i should map to
            mat2's node j. Thus, if this list is used to reorder a metamaterial,
            it should be used on mat2's reorder_nodes() function, not this material's.
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

        return pair
    

    def active_nodes(self) -> int:
        """
        Computes the number of nodes used by edges and faces in
        this metamaterial.

        Returns: `int`
            The number of nodes used by edges and faces in this
            metamaterial.
        """

        # Will store the active nodes
        nodes = set()

        # Finds nodes associated with edges
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                if self.has_edge(n1, n2):
                    nodes.add(n1)
                    nodes.add(n2)

        # Finds nodes associated with faces
        for n1 in range(NUM_NODES):
            for n2 in range(n1+1, NUM_NODES):
                for n3 in range(n2+1, NUM_NODES):
                    if self.has_face(n1, n2, n3):
                        nodes.add(n1)
                        nodes.add(n2)
                        nodes.add(n3)

        return len(nodes)
    

    def greedily_reorder_nodes(self, material: Self):

        # Computes the number of active nodes 
        nodes1 = self.active_nodes()
        nodes2 = material.active_nodes()

        # Finds the square distance between each node
        distances = {(n1,n2) : ((self.get_node_position(n1) - material.get_node_position(n2))**2).sum()
                        for n1 in range(nodes1)
                            for n2 in range(nodes2)}

        # Sorts the node pairs by increasing distance
        node_pairs = sorted([(n1,n2) for n1 in range(nodes1) for n2 in range(nodes2)], key=lambda x: distances[x])

        # Assigns the closest nodes to each other greedily
        assignments = {}
        for n1,n2 in node_pairs:

            # Skips already-seen nodes
            if n1 in assignments.values() or n2 in assignments:
                continue

            # Stores the assignment
            assignments[n2] = n1

        # Finds the leftover nodes in the other metamaterial
        leftovers = set()
        for n2 in range(nodes2):
            if n2 not in assignments:
                leftovers.add(n2)

        # Assigns the closest leftover nodes greedily
        leftover_assignments = {}   
        for n1,n2 in node_pairs:

            # Skips already-seen nodes
            if n2 not in leftovers or n1 in leftover_assignments.values() or n2 in leftover_assignments:
                continue

            # Stores the assignment
            leftover_assignments[n2] = n1

        # Creates the new nodes
        copy = self.copy()
        for n2 in leftovers:
            n1 = leftover_assignments[n2]
            node_pos = self.node_pos[n1*3 : n1*3+3] 
            copy.node_pos[nodes1*3 : nodes1*3+3] = node_pos
            for n3 in range(nodes1):
                if n1 == n3:
                    continue
                edge_index = edge_adj_index(nodes1,n3) * EDGE_BEZIER_COORDS
                copy.edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = copy.edge_params[edge_adj_index(n1,n3)*EDGE_BEZIER_COORDS:][:6]

            edge_index = edge_adj_index(n1,nodes1) * EDGE_BEZIER_COORDS
            copy.edge_params[edge_index : edge_index + EDGE_BEZIER_COORDS] = np.tile(self.get_node_position(n1), EDGE_BEZIER_POINTS)
            assignments[n2] = nodes1
            nodes1 += 1

        # Computes the node reordering
        node_reordering = [i for i in range(NUM_NODES)]
        for n1,n2 in assignments.items():
            node_reordering[n1] = n2

        # Reorders the nodes
        copy = copy.reorder_nodes(node_reordering)
        copy.edge_adj = material.edge_adj.copy()

        return copy


    def flatten_rep(self, pad_dim=False) -> torch.Tensor:
        """
        Computes the flattened array representation of the metamaterial.

        pad_dim: bool
            Whether an extra dimension will be added for padding to act as a batch of 1.
        """
        concatenation = np.concatenate((self.node_pos, self.edge_adj, self.edge_params, self.face_adj, self.face_params, np.array([self.thickness])))
        if pad_dim:
            concatenation = concatenation.reshape((1,concatenation.shape[0]))
        return torch.from_numpy(concatenation).type(torch.float32)


    def copy(self) -> Self:
        """
        Creates a copy of the Metamaterial.

        Returns: Metamaterial
            A copy of this metamaterial
        """

        # Copies the representation arrays
        material = Metamaterial(self.node_pos, self.edge_adj, self.edge_params, self.face_adj, self.face_params, thickness=self.thickness)

        # Copies the mirror transforms
        material.mirrors = self.mirrors.copy()

        # Copies the translate transforms
        material.translations = self.translations.copy()

        # Copies the displayed planes
        material.displayed_planes = self.displayed_planes.copy()

        return material
    
    
    def from_tensor(rep_tensor: torch.Tensor) -> Self:
        """
        Creates a Metamaterial from the given PyTorch tensor.

        rep_tensor: Tensor
            A tensor with the representation arrays concatenated together.
        """

        numpy_rep = rep_tensor.detach().reshape(REP_SIZE).numpy()

        # Stores the individual parts of the rep
        node_pos = numpy_rep[ : NODE_POS_SIZE]
        edge_adj = (numpy_rep[NODE_POS_SIZE : NODE_POS_SIZE + EDGE_ADJ_SIZE] > 0.5).astype(float)
        edge_params = numpy_rep[NODE_POS_SIZE + EDGE_ADJ_SIZE : NODE_POS_SIZE + EDGE_ADJ_SIZE + EDGE_PARAMS_SIZE]
        face_adj = (numpy_rep[NODE_POS_SIZE + EDGE_ADJ_SIZE + EDGE_PARAMS_SIZE : NODE_POS_SIZE + EDGE_ADJ_SIZE + EDGE_PARAMS_SIZE + FACE_ADJ_SIZE] > 0.5).astype(float)
        face_params = numpy_rep[NODE_POS_SIZE + EDGE_ADJ_SIZE + EDGE_PARAMS_SIZE + FACE_ADJ_SIZE : -1]
        thickness = numpy_rep[-1]

        return Metamaterial(node_pos, edge_adj, edge_params, face_adj, face_params, thickness=thickness)
    