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

