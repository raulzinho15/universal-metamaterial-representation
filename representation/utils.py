import numpy as np

def triangle_line_intersection(tri1: np.ndarray, tri2: np.ndarray, tri3: np.ndarray, line0: np.ndarray, line1: np.ndarray):
    """
    Checks whether the given line intersects the given triangle.

    tri1: ndarray
        A 1-d numpy array with the coordinates of the first vertex
        of the triangle.

    tri2: ndarray
        A 1-d numpy array with the coordinates of the second vertex
        of the triangle.

    tri3: ndarray
        A 1-d numpy array with the coordinates of the third vertex
        of the triangle.

    line0: ndarray
        A 1-d numpy array with the coordinates of the first vertex
        of the line.

    line1: ndarray
        A 1-d numpy array with the coordinates of the second vertex
        of the line.

    Returns: bool
        Whether the line intersects the triangle.
    """

    # Sets up the linear system
    A = np.stack((tri1-tri2, tri1-tri3, line1-line0), axis=1)
    b = tri1-line0

    # Solves the linear system
    try:
        bgt = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return False, float('inf')

    # Checks for an intersection
    beta, gamma, t = bgt
    return beta+gamma <= 1+1e-4 and beta >= -1e-4 and gamma >= -1e-4 and -1e-4 <= t <= 1+1e-4, t


def triangle_ray_intersection(tri1: np.ndarray, tri2: np.ndarray, tri3: np.ndarray, point: np.ndarray, ray: np.ndarray):
    """
    Checks whether the given line intersects the given triangle.

    tri1: ndarray
        A 1-d numpy array with the coordinates of the first vertex
        of the triangle.

    tri2: ndarray
        A 1-d numpy array with the coordinates of the second vertex
        of the triangle.

    tri3: ndarray
        A 1-d numpy array with the coordinates of the third vertex
        of the triangle.

    point: ndapointrray
        A 1-d numpy array with the coordinates of the point from 
        which the ray is shooting.

    ray: ndarray
        A 1-d numpy array with the coordinates of the ray shooting
        from the point.

    Returns: bool
        Whether the point-ray intersects the triangle.
    """

    # Sets up the linear system
    A = np.stack((tri1-tri2, tri1-tri3, ray), axis=1)
    b = tri1-point

    # Solves the linear system
    try:
        bgt = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return False, float('inf')

    # Checks for an intersection
    beta, gamma, t = bgt
    return beta+gamma <= 1+1e-4 and beta >= -1e-4 and gamma >= -1e-4 and t >= -1e-4, t


def is_above_triangle(tri1: np.ndarray, tri2: np.ndarray, tri3: np.ndarray, point: np.ndarray) -> bool:
    """
    Checks whether the given point is above the given triangle.

    tri1: ndarray
        A 1-d numpy array with the coordinates of the first vertex
        of the triangle.

    tri2: ndarray
        A 1-d numpy array with the coordinates of the second vertex
        of the triangle.

    tri3: ndarray
        A 1-d numpy array with the coordinates of the third vertex
        of the triangle.

    point: ndapointrray
        A 1-d numpy array with the coordinates of the point.

    Returns: bool
        Whether the point is above the triangle.
    """
    plane_normal = np.cross(tri2-tri1, tri3-tri1)
    return np.dot(plane_normal, point-tri1) > 0


def reachable_nodes(graph: dict[int, list[int]], starting_node: int) -> dict[int, tuple[int]]:
    
    # Stores the queue of nodes to check
    queue = [starting_node]

    # Stores the nodes that have been seen
    seen = set(queue)

    # Stores the paths seen so far
    paths = {starting_node : (starting_node,)}

    # Runs BFS
    while queue:

        # Stores the current node
        current = queue.pop(0)

        # Continues BFS to the current node's neighbors
        if current in graph:
            
            # Runs through each neighbor
            for neighbor in graph[current]:

                # Ignores already-seen neighbors
                if neighbor in seen:
                    continue

                # Stores info about this neighbor
                paths[neighbor] = paths[current] + (neighbor,)
                queue.append(neighbor)
                seen.add(neighbor)

    # Returns all paths to all connected nodes
    paths.pop(starting_node)
    return paths


def greedy_topology_match(start_adj_matrix: np.ndarray, target_adj_matrix: np.ndarray) -> np.ndarray:
    """
    Performs a greedy permutation of `start_adj_matrix` in such a way that
    every step tries to minimize the topology difference (via the edge connections)
    with the target adjacency matrix.

    start_adj_matrix: `np.ndarray`
        The starting adjacency matrix to be permuted greedily.
        Assumes the matrix contains both its upper and lower triangle values, and has
        identical values along the diagonal.

    target_adj_matrix: `np.ndarray`
        The target adjacency matrix to be permuted greedily into.
        Assumes the matrix contains both its upper and lower triangle values, and has
        identical values along the diagonal.

    Returns: `np.ndarray`
        The permuted adjacency matrix that was greedily matched to `target_adj_matrix`.
    """

    # Stores values for the function
    best_matrix = start_adj_matrix.copy()
    num_nodes = best_matrix.shape[0]

    # Continues trying until there are no more greedy choices
    while True:

        # Stores the current edge difference
        current = np.abs(best_matrix-target_adj_matrix).sum()

        # Stores the iteration's greediest step
        best_change = 0
        best_swap = None

        # Runs through each node pair
        for n1 in range(num_nodes):
            for n2 in range(n1+1, num_nodes):
                
                # Simulates this node pair swap
                matrix_copy = best_matrix.copy()
                matrix_copy[[n1,n2],:] = matrix_copy[[n2,n1],:]
                matrix_copy[:,[n1,n2]] = matrix_copy[:,[n2,n1]]

                # Checks if this swap is the best seen so far
                change = current - np.abs(matrix_copy-target_adj_matrix).sum()
                if change > best_change:
                    best_change = change
                    best_swap = (n1,n2)

        # Stop if no greedy swap was found
        if best_swap is None:
            break

        # Performs the best swap
        n1, n2 = best_swap
        best_matrix[[n1,n2],:] = best_matrix[[n2,n1],:]
        best_matrix[:,[n1,n2]] = best_matrix[:,[n2,n1]]

    return best_matrix


def graph_dict_from_adj_matrix(adj_matrix: np.ndarray) -> dict[int, list]:
    """
    Produces a dictionary describing the connections for the graph
    described in the given adjacency matrix.

    adj_matrix: `np.ndarray`
        The adjacency matrix for the graph. Only needs to have
        values on the upper triangle; the rest is ignored.

    Returns: `dict[int, list]`
        The dictionary for the graph describing its node connections.
        Namely, if `graph[i]` has `j`, then node `i` is connected to `j`.
    """

    # Stores values for the function
    num_nodes = adj_matrix.shape[0]
    graph_dict = {i : [] for i in range(num_nodes)}

    # Goes through each node pair
    for n1 in range(num_nodes):
        for n2 in range(n1+1, num_nodes):

            # Stores the graph connection
            if adj_matrix[n1,n2] == 1:
                graph_dict[n1].append(n2)
                graph_dict[n2].append(n1)

    return graph_dict


def find_connected_components(graph_dict: dict[int, list]) -> list[tuple[int]]:
    """
    Finds each connected component in the given graph, exluding
    nodes that are alone.

    graph_dict: `dict[int, list]`
        The dictionary for the graph describing its node connections.
        Namely, if `graph[i]` has `j`, then node `i` is connected to `j`.

    Returns: `list[tuple[int]]`
        A list of each collection of nodes that are in a connected component.
        Each tuple represents a separate connected component, in no particular
        order.
    """

    # Stores values for the function
    components = []
    nodes_seen = set()

    # Runs through each node
    for n in graph_dict:

        # Skips already-seen nodes
        if n in nodes_seen:
            continue
        nodes_seen.add(n)

        # Finds the other nodes in the connected component
        # starting at this node
        reachable = list(reachable_nodes(graph_dict, n).keys())

        # Skips trivial connected components
        if reachable is None:
            continue

        # Stores the connected component
        components.append((n,) + tuple(reachable))

        # Stores all the nodes that were seen
        for node in reachable:
            nodes_seen.add(node)

    return components
