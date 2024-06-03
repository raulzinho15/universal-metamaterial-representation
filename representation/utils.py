import numpy as np

def triangle_line_intersection(tri1: np.ndarray, tri2: np.ndarray, tri3: np.ndarray, line1: np.ndarray, line2: np.ndarray):
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

    line1: ndarray
        A 1-d numpy array with the coordinates of the first vertex
        of the line.

    line2: ndarray
        A 1-d numpy array with the coordinates of the second vertex
        of the line.

    Returns: bool
        Whether the line intersects the triangle.
    """

    # Creates column vectors
    tri1 = tri1.reshape((3, 1))
    tri2 = tri2.reshape((3, 1))
    tri3 = tri3.reshape((3, 1))
    line0 = line1.reshape((3, 1))
    lineD = (line2-line1).reshape((3, 1))

    # Sets up the linear system
    A = np.concatenate([tri1-tri2, tri1-tri3, lineD], axis=1)
    b = (tri1-line0)[:,0]

    # Solves the linear system
    try:
        bgt = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return False

    # Checks for an intersection
    beta, gamma, t = bgt[0], bgt[1], bgt[2]
    if beta+gamma <= 1 and beta >= 0 and gamma >= 0 and 0 <= t <= 1:
        return True
    return False
