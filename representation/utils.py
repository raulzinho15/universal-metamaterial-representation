import numpy as np

def triangle_line_intersection(tri1, tri2, tri3, line1, line2):
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
    tri1 = np.reshape(tri1, (3, 1))
    tri2 = np.reshape(tri2, (3, 1))
    tri3 = np.reshape(tri3, (3, 1))
    line0 = np.reshape(line1, (3, 1))
    lineD = np.reshape(line2-line1, (3, 1))

    # Sets up & solves the linear system
    A = np.concatenate([tri1-tri2, tri1-tri3, lineD], axis=1)
    b = np.reshape(tri1-line0, (3,))
    try:
        bgt = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return False

    # Checks for intersection
    beta, gamma, t = bgt[0], bgt[1], bgt[2]
    if beta+gamma <= 1 and beta >= 0 and gamma >= 0 and 0 <= t <= 1:
        return True
    return False
