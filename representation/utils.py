import numpy as np
import math


def greedy_topology_match(start_adj_matrix: np.ndarray, target_adj_matrix: np.ndarray) -> tuple[np.ndarray, list[tuple[int]]]:
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

    Returns: `np.ndarray`, `list[tuple[int]]`
        The first returned item is the permuted adjacency matrix that was greedily
        matched to `target_adj_matrix`.
        The second returned item is the sequence of swaps (each relative to the nodes
        at that point in time, not from the beginning) that led to the found matrix.
    """

    # Stores values for the function
    best_matrix = start_adj_matrix.copy()
    best_swaps = []
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
        best_swaps.append(best_swap)
        n1, n2 = best_swap
        best_matrix[[n1,n2],:] = best_matrix[[n2,n1],:]
        best_matrix[:,[n1,n2]] = best_matrix[:,[n2,n1]]

    return best_matrix, best_swaps


LINE_NORMAL_POINT3 = np.array([np.pi, np.e, np.sqrt(2)])
def find_line_normals(point1: np.ndarray, point2: np.ndarray) -> tuple[np.ndarray]:
    """
    Finds consistent normals for the given line.

    point1: np.ndarray
        The start point on the line. Assumes in the unit cube.

    point2: np.ndarray
        The last point on the line. Assumes in the unit cube.

    Returns: tuple[np.ndarray]
        The two (normalized) normals of the line. Finds
        these deterministically for any given line, assuming
        point1 and point2 are ordered consistently.
    """

    # Finds the plane vectors
    vector1 = point2 - point1
    vector2 = LINE_NORMAL_POINT3 - point1

    # Finds the plane normal
    normal1 = np.cross(vector1, vector2)
    normal1 /= np.linalg.norm(normal1)

    # Finds the second normal
    normal2 = np.cross(vector1, normal1)
    normal2 /= np.linalg.norm(normal2)

    return normal1, normal2


def rotate_around_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Uses Rodrigues' rotation formula to rotate the given vector
    by the given amount around the given axis.

    vector: np.ndarray
        The vector to be rotated.

    axis: np.ndarray
        The axis around which the vector will be rotated.
        Assumed to be a unit vector.

    angle: float
        The angle by which the given vector will be rotated.
        Assumed to be in radians.

    Returns: np.ndarray
        The rotated vector.
    """

    return (
        vector * np.cos(angle) +
        np.cross(axis, vector) * np.sin(angle) +
        axis * np.dot(axis, vector) * (1 - np.cos(angle))
    )