import numpy as np


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
