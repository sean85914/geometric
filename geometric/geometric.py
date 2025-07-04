import numpy as np
import warnings

from .relation_enum import (PointTriangleEnum, PointCircleEnum, PointShapeEnum)


XY_PLANE = [0.0, 0.0, 1.0, 0.0]
YZ_PLANE = [1.0, 0.0, 0.0, 0.0]
ZX_PLANE = [0.0, 1.0, 0.0, 0.0]


def norm(vector):
    '''Calculate the norm of a given vector.

    Arguments:
        vector (list or array-like): The vector to compute the norm

    Return:
        float: The norm of the vector
    '''
    return np.linalg.norm(vector)


def is_zero_vector(vector):
    '''Check if a given vector is a zero vector within a specified tolerance.

    Arguments:
        vector (list or array-like): The vector to check

    Return:
        bool: True if the vector is a zero vector, False otherwise
    '''
    return np.isclose(norm(vector), 0.0, atol=1e-5)


def unit_vector(vector):
    '''Calculate the normalized vector of a given vector.

    Arguments:
        vector (list or array-like): The vector to normalized

    Return:
        numpy.ndarray: The unit vector of the given vector

    Raise:
        AssertionError: If the input vector is a zero vector
    '''
    assert not is_zero_vector(vector), 'input vector is a zero vector'
    return vector / np.linalg.norm(vector)


def is_on_axis(vector):
    '''Check if a given vector lies on one of the coordinate axes.

    Arguments:
        vector (list or array-like): The vector to check

    Return:
        tuple: A tuple containing

        - bool: True if the vector lies on one of the coordinate axes, False otherwise.
        - int: The index of the axis if the vector lies on an axis, -1 otherwise.

    Raise:
        AssertionError: If the input vector is a zero vector

    Note:
        0 for X, 1 for Y and 2 for Z
    '''
    unit = unit_vector(vector)
    for idx, component in enumerate(unit):
        if np.isclose(abs(component), 1.0, atol=1e-5):
            return True, idx
    return False, -1


def middle_point(p1, p2):
    '''Calculate the middle point between two points in the same dimensional space.

    Arguments:
        p1 (list or array-like): The first point.
        p2 (list or array-like): The second point.

    Returns:
        numpy.ndarray: The coordinates of the middle point.

    Raises:
        AssertionError: If the two points do not have the same dimensions.
    '''
    assert len(p1) == len(p2), 'Two points do not have the same dimensions'
    return (np.array(p1) + np.array(p2)) / 2


def average_point(*p, **kwargs):
    '''Calculate the weighted average point of given points.

    Arguments:
        p (array-like): A variable number of points
        kwargs (dict): Optional keyword arguments

            - weights (array-like, optional): The weights associated with each point. Default is None

    Returns:
        numpy.ndarray: The coordinates of the average point.

    Raises:
        ValueError: If the dimensions of the input points or weights do not match,
                    or if weights contain any invalid values.
    '''
    p = np.array(p)
    return np.average(p, weights=kwargs.get('weights', None), axis=0)


def distance_between_points(p1, p2):
    '''Calculate the Euclidean distance between two points.

    Arguments:
        p1 (list or array-like): The first point
        p2 (list or array-like): The second point

    Returns:
        float: The Euclidean distance between the two points

    Raises:
        AssertionError: If the two points do not have the same dimensions
    '''
    assert len(p1) == len(p2), 'Two points do not have the same dimensions'
    return np.linalg.norm(np.array(p1) - np.array(p2))


def line_from_point_vector(point, vector):
    '''Generate the equation of a line in 2D or 3D space given a point on the line and its direction vector.

    Arguments:
        point (list or array-like): A point on the line
        vector (list or array-like): The direction vector of the line

    Returns:
        - 2D
            - list: The coefficients of the line equation, :math:`ax + by + c = 0`
        - 3D
            - list: The point which on the line
            - list: Normalized direction vector

    Raises:
        AssertionError: If the dimensions of the point and vector do not match the expected dimensions (2 or 3).
    '''
    def two_dim():
        assert len(point) == len(vector) == 2
        a, b = point
        s, t = unit_vector(vector)
        const = np.dot([t, s], [a, -b])
        return [t, -s, -const]

    def three_dim():
        assert len(point) == len(vector) == 3
        return [point, unit_vector(vector).tolist()]

    if len(point) == 2:
        return two_dim()
    elif len(point) == 3:
        return three_dim()
    assert False, 'Point and vector dimensions must be either 2 or 3'


def line_from_two_points(p1, p2):
    '''Generate the equation of a line in 2D or 3D space passing through two given points.

    Arguments:
        p1 (list or array-like): Coordinates of the first point
        p2 (list or array-like): Coordinates of the second point

    Returns:
        - 2D
            - list: The coefficients of the line equation, :math:`ax + by + c = 0`
        - 3D
            - list: The point which on the line
            - list: Normalized direction vector

    Raise:
        AssertionError: If the dimensions of the points do not match the expected dimensions (2D or 3D).
    '''
    def two_dim():
        assert len(p1) == len(p2) == 2
        a1, b1 = p1
        a2, b2 = p2
        vector = np.array([b1 - b2, a2 - a1], dtype=float)
        vector = unit_vector(vector)
        const = np.dot(vector, p1)
        return [vector[0], vector[1], -const]

    def three_dim():
        assert len(p1) == len(p2) == 3
        vector = np.array(p2, dtype=float) - np.array(p1, dtype=float)
        vector = unit_vector(vector)
        a1, b1, c1 = p1
        return [[a1, b1, c1], vector.tolist()]

    if len(p1) == 2:
        return two_dim()
    elif len(p1) == 3:
        return three_dim()
    assert False, 'Points dimensions must be either 2 or 3'


def random_point_on_line(line):
    '''Generate a random point on a line in 2D or 3D space.

    The function accepts two formats for the input `line`:

    (1) 2D line
        A list or array-like of three coefficients [a, b, c], representing the line equation :math:`ax + by + c = 0`.
    (2) 3D line: A list or array-like of two elements:

        - `line[0]`: A 3D point on the line.
        - `line[1]`: A normalized direction vector of the line.

    Arguments:
        line (list or array-like): Representation of the line, either 2D ([a, b, c]) or 3D ([point, vector]).

    Returns:
        numpy.ndarray: Coordinates of a random point on the line.

    Raises:
        AssertionError: If the dimensions of the line do not match the expected dimensions (2D or 3D).
    '''
    def two_dim():
        length = norm(line[:2])
        normalized = np.array(line) / length
        on_axis, index = is_on_axis(normalized[:2])
        r = np.random.uniform(-1., 1.)
        if on_axis:
            p = [0., 0.]
            random_index = 0 if index == 1 else 1
            p[random_index] = r
            p[index] = -normalized[2] / normalized[index]
        else:
            y = -(normalized[0] * r + normalized[2]) / normalized[1]
            p = [r, y]
        return np.array(p)

    def three_dim():
        t = np.random.uniform(-1., 1.)
        return np.array(line[0]) + t * np.array(line[1])

    if len(line) == 3:
        return two_dim()
    elif len(line) == 2 and len(line[0]) == len(line[1]) == 3:
        return three_dim()
    assert False, 'Line dimensions must be either 2D or 3D'


def is_point_on_line(point, line):
    '''Check if a point lies on a given line in 2D or 3D space.

    Arguments:
        point (list or array-like): Coordinates of the point to be checked.
        line (list or array-like): Representation of the line:

            (1) 2D: the coefficients of the line equation, :math:`ax + by + c = 0`

            (2) 3D: ``[[x1, y1, z1], [vx, vy, vz]]`` where ``[x1, y1, z1]`` is a point on the line
                and ``[vx, vy, vz]`` is the direction vector of the line

    Returns:
        bool: True if the point lies on the line, False otherwise.

    Raises:
        AssertionError: If the dimensions of the point and line do not match the expected dimensions
    '''
    def two_dim():
        x, y = point
        return np.isclose(np.dot(line, [x, y, 1.0]), 0, atol=1e-5)

    def three_dim():
        line_point = line[0]
        vector = line[1]
        direction = np.array(point) - np.array(line_point)
        if np.isclose(np.linalg.norm(direction), 0, atol=1e-5):
            return True
        angle = angle_between_vectors(direction, vector)
        return np.isclose(angle, 0, atol=1e-5) or np.isclose(angle, np.pi, atol=1e-5)

    if len(line) == 3 and len(point) == 2:
        return two_dim()
    elif len(line) == 2 and len(line[0]) == 3 and len(point) == 3:
        return three_dim()
    assert False


def plane_from_three_points(p1, p2, p3):
    '''Calculate the plane equation from three points in 3D space.

    Arguments:
        p1 (list or array-like): Coordinates of the first point
        p2 (list or array-like): Coordinates of the second point
        p3 (list or array-like): Coordinates of the third point

    Returns:
        list: The coefficients of the plane equation in the form :math:`ax + by + cz + d = 0`

    Raises:
        AssertionError: If the dimensions of the points are not equal to 3,
                        or if the points are collinear.
    '''
    assert len(p1) == len(p2) == len(p3) == 3, 'Points must be in 3D'
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    assert not np.isclose(np.linalg.norm(normal), 0, atol=1e-5), 'Given points are collinear'
    normal = unit_vector(normal)
    const = np.dot(normal, p1)
    nx, ny, nz = normal
    return [nx, ny, nz, -const]


def plane_from_point_vector(point, vector):
    '''Generate the plane equation in 3D space given a point on the plane and its normal vector.

    Arguments:
        point (list or array-like): A point on the plane
        vector (list or array-like): The normal vector to the plane

    Returns:
        list: The coefficients of the plane equation in the form :math:`ax + by + cz + d = 0`

    Raises:
        AssertionError: If the dimensions of the point and vector are not equal to 3.
    '''
    assert len(point) == len(vector) == 3, 'Invalid point or vector'
    vector = unit_vector(vector)
    return np.array(np.append(vector, -np.dot(vector, point))).tolist()


def plane_from_noisy_data(points):
    '''Estimate the best-fitting plane from a set of noisy 3D points using least squares method.

    The plane is estimated in the form: :math:`ax + by + cz + d = 0`, with :math:`\\sqrt{a^2 + b^2 + c^2} = 1`.

    Arguments:
        points (array-like): A list or array of shape ``(N, 3)``, where N > 3, containing 3D points.

    Returns:
        list: Normalized plane coefficients ``[a, b, c, d]`` such that :math:`ax + by + cz + d = 0`.

    Raises:
        AssertionError: If the input does not contain at least 4 points or if points are not 3D.
    '''
    points = np.array(points)
    assert points.shape[1] == 3
    assert points.shape[0] > 3
    A = np.ones((points.shape[0], 3))
    b = np.ones((points.shape[0], 1))
    for i in range(points.shape[0]):
        A[i, 0] = points[i, 0]
        A[i, 1] = points[i, 1]
        b[i, 0] = points[i, 2]
    x = np.linalg.inv(A.T @ A) @ A.T @ b
    a, b, c = x.flatten()
    # z = ax + by + c, or ax + by - z + c = 0
    return (np.array([a, b, -1, c]) / norm([a, b, -1])).tolist()


def random_point_on_plane(plane):
    '''Generate a random point on a plane in 3D space.

    Arguments:
        plane (list or array-like): the coefficients of the plane equation in the form :math:`ax + by + cz + d = 0`

    Returns:
        numpy.ndarray: Coordinates of a random point on the plane

    Raises:
        AssertionError: If the input plane is invalid
    '''
    assert len(plane) == 4, 'Input plane is invalid'
    length = norm(plane[:3])
    plane = np.array(plane) / length
    on_axis, index = is_on_axis(plane[:3])
    r1 = np.random.uniform(-1., 1.)
    r2 = np.random.uniform(-1., 1.)
    p = [0.0] * 3
    zero_index = np.where(plane[:3] == 0)[0]
    if len(zero_index) == 0:
        p[0] = r1
        p[1] = r2
        p[2] = -(plane[0] * r1 + plane[1] * r2 + plane[3]) / plane[2]
    else:
        index = zero_index[0]
        index_1 = (index + 1) % 3
        index_2 = (index + 2) % 3
        p[index] = r1
        p[index_1] = r2
        p[index_2] = -(plane[3] + plane[index_1] * r2) / plane[index_2]
    return p


def perpendicular_bisector(p1, p2):
    '''Calculate the perpendicular bisector of a line segment in 2D or 3D space.

    The function determines the perpendicular bisector:

    - In **2D**, it returns the line coefficients [a, b, c] of the equation: :math:`ax + by + c = 0`.
    - In **3D**, it returns the plane coefficients [a, b, c, d] of the equation: :math:`ax + by + cz + d = 0`.

    Arguments:
        p1 (list or array-like): Coordinates of the first endpoint of the line segment.
        p2 (list or array-like): Coordinates of the second endpoint of the line segment.

    Returns:
        list: Coefficients of the perpendicular bisector line (2D) or plane (3D).

    Raises:
        AssertionError: If the dimensions of the points do not match the expected dimensions (2D or 3D).
    '''
    def two_dim():
        mp = middle_point(p1, p2)
        vector = np.array(p2) - np.array(p1)
        vector = [vector[1], -vector[0]]
        return line_from_point_vector(mp, vector)

    def three_dim():
        mp = middle_point(p1, p2)
        vector = np.array(p2) - np.array(p1)
        return plane_from_point_vector(mp, vector)

    if len(p1) == 2:
        return two_dim()
    elif len(p1) == 3:
        return three_dim()
    assert False, 'Points dimensions must be either 2D or 3D'


def angle_bisector_line_from_two_lines(line_1, line_2):
    '''Calculate the two angle bisector lines from two input lines in 2D or 3D space.

    - In 2D: Each line is represented by coefficients [a, b, c] of the line equation :math:`ax + by + c = 0`.
    - In 3D: Each line is represented by a list [point, direction_vector].

    Arguments:
        line_1 (list or array-like): First line, either 2D coefficients or 3D [point, vector].
        line_2 (list or array-like): Second line, same format as `line_1`.

    Returns:
        tuple: A pair of bisector lines.

            * In 2D, each is a list of coefficients [a, b, c].
            * In 3D, each is a list [point, direction_vector].

    Raises:
        AssertionError: If the input lines are invalid or do not intersect (e.g., parallel lines).
    '''
    def two_dim():
        try:
            _ = intersection_between_lines(line_1, line_2)
        except AssertionError:
            raise AssertionError('Two lines are parallel') from None
        unit_line_1 = np.array(line_1) / norm(line_1[:2])
        unit_line_2 = np.array(line_2) / norm(line_2[:2])
        bisector_line_1 = unit_vector(unit_line_1 + unit_line_2)
        bisector_line_2 = unit_vector(unit_line_1 - unit_line_2)
        return bisector_line_1, bisector_line_2

    def three_dim():
        try:
            point = intersection_between_lines(line_1, line_2)
        except AssertionError:
            raise AssertionError('Two lines are not intersect at one point') from None
        _1 = np.array(line_1[1])
        _2 = np.array(line_2[1])
        bisector_line_1 = unit_vector(_1 + _2)
        bisector_line_2 = unit_vector(_1 - _2)
        return line_from_point_vector(point, bisector_line_1), \
            line_from_point_vector(point, bisector_line_2),

    if len(line_1) == 3 and len(line_2) == 3:
        return two_dim()
    elif np.array(line_1).shape == (2, 3) and np.array(line_2).shape == (2, 3):
        return three_dim()
    else:
        assert False, 'Invalid lines'


def angle_bisector_plane_from_two_lines(line_1, line_2):
    '''Compute the two angle bisector planes formed by two intersecting 3D lines.

    Each line is defined by [point, direction_vector].

    Arguments:
        line_1 (list): First 3D line as [point, direction_vector].
        line_2 (list): Second 3D line as [point, direction_vector].

    Returns:
        tuple: A pair of planes, each represented by coefficients [a, b, c, d] of the plane equation
               :math:`ax + by + cz + d = 0`.

    Raises:
        AssertionError: If input lines are invalid or do not intersect.
    '''
    assert np.array(line_1).shape == (2, 3) and np.array(line_2).shape == (2, 3), \
           'Invalid lines'
    try:
        point = intersection_between_lines(line_1, line_2)
    except AssertionError:
        raise AssertionError('Two lines are not intersect at one point') from None
    _1 = np.array(line_1[1])
    _2 = np.array(line_2[1])
    n = np.cross(_1, _2)
    bisector_line_1 = _1 + _2
    bisector_line_2 = _1 - _2
    bisector_plane_1 = unit_vector(np.cross(n, bisector_line_1))
    bisector_plane_2 = unit_vector(np.cross(n, bisector_line_2))
    return plane_from_point_vector(point, bisector_plane_1), \
        plane_from_point_vector(point, bisector_plane_2)


def angle_bisector_plane_from_two_planes(plane_1, plane_2):
    '''Compute the two angle bisector planes formed by the intersection of two planes.

    The result is two new planes whose normal vectors bisect the angle between the original planes.

    Arguments:
        plane_1 (list): First plane coefficients [a, b, c, d] of the equation :math:`ax + by + cz + d = 0`.
        plane_2 (list): Second plane coefficients [a, b, c, d].

    Returns:
        tuple: A pair of normalized plane coefficients [a, b, c, d].

    Raises:
        AssertionError: If the input planes are invalid or do not intersect along a line.
    '''
    assert len(plane_1) == len(plane_2) == 4, 'Invalid planes'
    try:
        _ = line_from_planes(plane_1, plane_2)
    except AssertionError:
        raise AssertionError('Two planes are not intersect at a line')
    unit_plane_1 = np.array(plane_1) / norm(plane_1[:3])
    unit_plane_2 = np.array(plane_2) / norm(plane_2[:3])
    bisector_plane_1 = unit_vector(unit_plane_1 + unit_plane_2)
    bisector_plane_2 = unit_vector(unit_plane_1 - unit_plane_2)
    return bisector_plane_1, bisector_plane_2


def nearest_point(point, vector, target_point):
    '''Given a point and a vector, as well as another point, return the
    nearest point on the line passing through the point in the direction
    of the vector and the other point.

    Arguments:
        point (list or array-like): Coordinates of the point through which the line passes
        vector (list or array-like): Direction vector of the line
        target_point (list or array-like): Coordinates of the target point

    Return:
        numpy.array: the nearest point on the line

    See Also:
        `project_point_on_line <#geometric.project_point_on_line>`_
    '''
    assert len(point) == len(vector) == len(target_point)
    norm = np.power(np.linalg.norm(vector), 2)
    const = np.dot(np.array(target_point) - np.array(point), np.array(vector)) / norm
    nearest_point = np.array(point) + const * np.array(vector)
    return nearest_point


def nearest_distance(point, vector, target_point):
    '''Given a point and a vector, as well as another point, return the
    shortest distance between the line passing through the point in the
    direction of the vector and the other point.

    Arguments:
        point (list or array-like): Coordinates of the point through which the line passes
        vector (list or array-like): Direction vector of the line
        target_point (list or array-like): Coordinates of the target point

    Return:
        float: The shortest distance between the line and the target point
    '''
    n_point = nearest_point(point, vector, target_point)
    return distance_between_points(n_point, target_point)


def angle_between_vectors(v1, v2, degrees=False):
    '''Calculate the angle between two vectors in radians or degrees.

    Arguments:
        v1 (list or array-like): First vector
        v2 (list or array-like): Second vector
        degrees (bool, optional): If True, return the angle in degrees. Default is False (radians).

    Returns:
        float: Angle between v1 and v2

    Raises:
        AssertionError: If the dimensions of v1 and v2 are not equal,
                        or if both vectors are zero vectors.

    Note:
        Return value will be in range [0, :math:`\\pi`]
    '''
    assert len(v1) == len(v2), 'Vectors must have the same dimension'
    assert not (is_zero_vector(v1) and is_zero_vector(v2)), 'Vectors must not both be zero vectors'
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(min(max(c, -1), 1))
    if degrees:
        return np.degrees(theta)
    return theta


def orthogonal_vector(v):
    '''Return a unit vector (randomly, for n >= 3) that is orthogonal to the input vector.

    Arguments:
        v (list or array-like): Input vector

    Returns:
        numpy.ndarray: Unit vector that is orthogonal to the input vector

    Raises:
        AssertionError: If input vector is 1D
        AssertionError: If input is zero vector
    '''
    assert len(v) > 1, 'Input vector should at least with 2D'
    assert norm(v) > 0, 'Input should not be zero vector'
    if len(v) == 2:
        v1, v2 = v
        return unit_vector([v2, -v1])
    result = np.random.rand(len(v))
    acc = np.dot(v[:-1], result[:-1])
    result[-1] = -acc / v[-1]
    return unit_vector(result)


def vector_projection(v1, v2):
    '''Project a vector, v1, onto another vector, v2.

    Arguments:
        v1 (list or array-like): Vector to be projected
        v2 (list or array-like): Vector onto which v1 is projected

    Return:
        numpy.array: Projection of v1 onto v2

    Raises:
        AssertionError: If the dimensions of v1 and v2 are not equal,
                        or if v2 is a zero vector.
    '''
    assert len(v1) == len(v2), 'Vectors must have the same dimension'
    assert not is_zero_vector(v2), 'v2 must not be a zero vector'
    norm_square = np.power(np.linalg.norm(v2), 2)
    dot = np.dot(v1, v2)
    return dot / norm_square * np.array(v2)


def project_vector_on_plane(v, normal):
    '''Project a vector, v, onto a plane represented by its normal vector.

    Arguments:
        v (list or array-like): Vector to be projected.
        normal (list or array-like): Normal vector of the plane

    Return:
        numpy.array: Projection of v onto the plane

    Raises:
        AssertionError: If the dimensions of v and normal are not equal, or if they are not in 3D space
    '''
    assert len(v) == len(normal) == 3, 'Vectors must be in 3D space'
    v_parallel = vector_projection(v, normal)
    return v - v_parallel


def is_point_on_plane(point, plane):
    '''Check if a point lies on a given plane in 3D space.

    Arguments:
        point (list or array-like): coordinates of the point to be checked
        plane (list or array-like): representation of the plane as [a, b, c, d]
                                    where the plane equation is ax + by + cz + d = 0.

    Returns:
        on_plane (bool): True if the point lies on the plane, False otherwise

    Raises:
        AssertionError: if the dimensions of the plane and point do not match the expected dimensions
    '''
    assert len(plane) == 4 and len(point) == 3
    x, y, z = point
    return np.isclose(np.dot(plane, [x, y, z, 1.0]), 0.0, atol=1e-5)


def is_line_on_plane(line, plane):
    '''Check if a line lies on a given plane in 3D space.

    Arguments:
        line (list or array-like): representation of the line as ``[[x1, y1, z1], [vx, vy, vz]]``
                                   where ``[x1, y1, z1]`` is a point on the line and
                                   ``[vx, vy, vz]`` is the direction vector of the line
        plane (array-like): representation of the plane as [a, b, c, d]
                            where the plane equation is :math:`ax + by + cz + d = 0`

    Returns:
        on_plane (bool): True if the line lies on the plane, False otherwise

    Raises:
        AssertionError: if the dimensions of the line and plane do not match the expected dimensions.
    '''
    assert np.array(line).shape == (2, 3) and len(plane) == 4
    point, vector = line
    if np.isclose(np.dot(vector, plane[:3]), 0.0, atol=1e-5) and is_point_on_plane(point, plane):
        return True
    return False


def project_point_on_line(p, line):
    '''Project a point onto a line in 2D or 3D space based on inputs.

    Arguments:
        p (list or array-like): The point to be projected
        line (list or array-like): Representation of the line:

            (1) 2D: the coefficients of the line equation, :math:`ax + by + c = 0`
            (2) 3D: ``[[x1, y1, z1], [vx, vy, vz]]`` where ``[x1, y1, z1]`` is a point on the line
                  and ``[vx, vy, vz]`` is the direction vector of the line

    Returns:
        numpy.array: The projected point on the line.

    Raises:
        AssertionError: If the dimensions of `p` and `line` do not match the expected dimensions.

    See Also:
        `nearest_point <#geometric.nearest_point>`_
    '''
    def two_dim():
        assert len(line) == 3
        x, y = p
        a, b, c = line
        norm_square = np.power(np.linalg.norm([a, b]), 2)
        d = -b * x + a * y
        x_p = (-b * d - a * c) / norm_square
        y_p = (a * d - b * c) / norm_square
        return np.array([x_p, y_p])

    def three_dim():
        assert np.array(line).shape == (2, 3)
        point, vector = line
        return nearest_point(point, vector, p)

    if len(p) == 2:
        return two_dim()
    elif len(p) == 3:
        return three_dim()
    assert False


def project_point_on_plane(p, plane):
    '''Project a point onto a given plane in 3D space.

    Arguments:
        p (list or array-like): Coordinates of the point to be projected
        plane (list or array-like): Representation of the plane as [a, b, c, d]
                                    where the plane equation is :math:`ax + by + cz + d = 0`

    Returns:
        numpy.array: Coordinates of the projected point on the plane.

    Raises:
        AssertionError: If the dimensions of the point and plane do not match the expected dimensions.
    '''
    assert len(p) == 3 and len(plane) == 4
    x, y, z = p
    a, b, c, d = plane
    norm_square = np.power(np.linalg.norm([a, b, c]), 2)
    t = - np.dot(plane, [x, y, z, 1]) / norm_square
    x_p = x + a * t
    y_p = y + b * t
    z_p = z + c * t
    return np.array([x_p, y_p, z_p])


def project_line_on_plane(line, plane):
    '''Project a line onto a given plane in 3D space.

    Arguments:
        line (list or array-like): Representation of the line as ``[[x1, y1, z1], [vx, vy, vz]]``
                                   where ``[x1, y1, z1]`` is a point on the line and
                                   ``[vx, vy, vz]`` is the direction vector of the line
        plane (list or array-like): Representation of the plane as [a, b, c, d]
                                    where the plane equation is :math:`ax + by + cz + d = 0`

    Returns:
        list: a list containing the projected point on the plane and the projected direction vector on the plane.

    Raises:
        AssertionError: if the dimensions of the line and plane do not match the expected dimensions.
    '''
    assert np.array(line).shape == (2, 3) and len(plane) == 4
    point, vector = line
    pp = project_point_on_plane(point, plane)
    pv = project_vector_on_plane(vector, plane[:3])
    return [pp, pv]


def distance_point_to_line(p, line):
    '''Calculate the distance from a point to a line in 2D space.

    Arguments:
        p (list or array-like): A single point ``[x, y]`` or multiple points ``[[x1, y1], [x2, y2], ...]``.
        line (list or array-like): Coefficients of the line equation ax + by + c = 0

    Returns:
        float or np.ndarray: Distance (if input is a single point) or an array of distances (if multiple points).

    Raises:
        AssertionError: If the dimensions of the point and line do not match the expected dimensions.

    .. todo::
        Should also supports 3D.
    '''
    p_ = np.atleast_2d(p)
    if p_.shape[0] == 1:
        p_project = project_point_on_line(p, line)
        return np.linalg.norm(p_project - np.array(p))
    else:
        ext_p = np.ones((p_.shape[0], 3))
        ext_p[:, :2] = p_
        num = ext_p @ np.array(line)
        den = norm(line[:2]) ** 2
        return np.abs(num) / den


def distance_point_to_plane(p, plane):
    '''Calculate the distance from a point to a plane in 3D space.

    Arguments:
        p (list or array-like): A single point ``[x, y, z]`` or multiple points ``[[x1, y1, z1], [x2, y2, z2], ...]``.
        plane (list or array-like): Representation of the plane as ``[a, b, c, d]``
                                    where the plane equation is :math:`ax + by + cz + d = 0`

    Returns:
        float or np.ndarray: Distance (if input is a single point) or an array of distances (if multiple points).

    Raises:
        AssertionError: If the dimensions of the point and plane do not match the expected dimensions
    '''
    p_ = np.atleast_2d(p)
    if p_.shape[0] == 1:
        p_project = project_point_on_plane(p, plane)
        return np.linalg.norm(p_project - np.array(p))
    else:
        ext_p = np.ones((p_.shape[0], 4))
        ext_p[:, :3] = p_
        num = ext_p @ np.array(plane)
        den = norm(plane[:3]) ** 2
        return np.abs(num) / den


def intersection_between_lines(line_1, line_2):
    '''Find the intersection point between two lines.

    Arguments:
        line_1 (list or array-like): the first line. Representation of the line:

            (1) 2D: The coefficients of the line equation :math:`ax + by + c = 0`
            (2) 3D: ``[[x1, y1, z1], [vx, vy, vz]]`` where ``[x1, y1, z1]`` is a point on the line
                  and ``[vx, vy, vz]`` is the direction vector of the line
        line_2 (list or array-like): the second line, the representation is the same as `line_1`

    Returns:
        numpy.ndarray: The intersection point

    Raises:
        AssertionError: If the lines are parallel (in 2D) or do not intersect (in 3D).
    '''
    def two_dim():
        assert len(line_1) == len(line_2) == 3
        a1, b1, c1 = line_1
        a2, b2, c2 = line_2
        den = np.linalg.det([[a1, b1], [a2, b2]])
        assert not np.isclose(den, 0, atol=1e-5)
        num_x = np.linalg.det([[-c1, b1], [-c2, b2]])
        num_y = np.linalg.det([[a1, -c1], [a2, -c2]])
        return np.array([num_x / den, num_y / den])

    def three_dim():
        assert np.array(line_1).shape == (2, 3)
        assert np.array(line_2).shape == (2, 3)
        (x1, y1, z1), (a1, b1, c1) = line_1
        (x2, y2, z2), (a2, b2, c2) = line_2
        A = np.array([
            [a1, -a2],
            [b1, -b2],
            [c1, -c2]
        ])
        b = np.array([
            [x2 - x1],
            [y2 - y1],
            [z2 - z1]
        ])
        if np.linalg.matrix_rank(A) < 2:
            assert False
        lstsq_result = np.linalg.lstsq(A, b, rcond=None)
        t, s = lstsq_result[0]
        residual = lstsq_result[1][0]
        if np.isclose(residual, 0.0, atol=1e-5):
            return (np.array([x1, y1, z1]) + np.array([a1, b1, c1]) * t).tolist()
        assert False, 'Two lines are not intersect'

    if len(line_1) == 3:
        return two_dim()
    elif len(line_1) == 2:
        return three_dim()
    assert False


def intersection_between_line_segments(line_1_points, line_2_points):
    '''Find the intersection point between two line segments, which are determined by their end-points.
    If this two line segments intersect, returns the intersection point; otherwise, an array filled with
    ``NaN`` (not a number) is returned.

    Arguments:
        line_1_points (list or array-like): Two points defining the first line segment
        line_2_points (list or array-like): Two points defining the second line segment

    Returns:
        numpy.ndarray: The intersection point if the line segments intersect within their bounds.
                       Otherwise, an array of NaNs.

    Raises:
        AssertionError: If the input points are not in the correct shape.
    '''
    assert np.array(line_1_points).shape[0] == 2
    assert np.array(line_2_points).shape[0] == 2
    dim = len(line_1_points[0])
    line_1 = line_from_two_points(line_1_points[0], line_1_points[1])
    line_2 = line_from_two_points(line_2_points[0], line_2_points[1])
    try:
        point = intersection_between_lines(line_1, line_2)
        t = (point[0] - line_1_points[0][0]) / (line_1_points[1][0] - line_1_points[0][0])
        s = (point[0] - line_2_points[0][0]) / (line_2_points[1][0] - line_2_points[0][0])
        if 0 <= t <= 1 and 0 <= s <= 1:
            return point
        return np.array([float('nan')] * dim)
    except AssertionError:
        return np.array([float('nan')] * dim)


def line_from_planes(plane_1, plane_2):
    '''Calculate the intersection line of two planes.

    Arguments:
        plane_1 (list or array-like): Representation of the first plane as ``[a, b, c, d]``
                                      where the plane equation is :math:`ax + by + cz + d = 0`
        plane_2 (list or array-like): Same as `plane_1`

    Returns:
        list: a list containing two elements:
              - A point ``[x, y, z]`` on the line of intersection
              - A direction vector ``[i, j, k]`` of the line of intersection

    Raises:
        AssertionError: if the input planes do not define a unique line (i.e., if they are parallel or coincide).
    '''
    assert len(plane_1) == 4 and len(plane_2) == 4
    normal_1 = np.array(plane_1)[:3]
    normal_2 = np.array(plane_2)[:3]
    vector = np.cross(normal_1, normal_2)
    assert not np.isclose(np.linalg.norm(vector), 0.0, atol=1e-5)
    vector = unit_vector(vector)
    if is_on_axis(unit_vector(normal_1))[0] and is_on_axis(unit_vector(normal_2))[0]:
        point = [0.0] * 3
        axis_1 = is_on_axis(normal_1)[1]
        axis_2 = is_on_axis(normal_2)[1]
        point[axis_1] = -plane_1[3] / normal_1[axis_1]
        point[axis_2] = -plane_2[3] / normal_2[axis_2]
        return [point, vector.tolist()]

    for i in range(3):
        line_1 = list(plane_1)
        line_2 = list(plane_2)
        line_1.pop(i)
        line_2.pop(i)
        if np.isclose(np.linalg.det([line_1[:2], line_2[:2]]), 0.0, atol=1e-5):
            continue
        indices = [0, 1, 2]
        indices.pop(i)
        point = [0.0] * 3
        _1, _2 = intersection_between_lines(line_1, line_2)
        point[indices[0]] = _1
        point[indices[1]] = _2
        return [point, vector.tolist()]

    assert False


def line_from_noisy_data(points):
    '''Estimate the best-fit line from noisy points in 2D or 3D space.

    - In **2D**, the line is fitted using least squares to minimize vertical error from the model :math:`y = ax + b`.
      The result is returned in the form of normalized coefficients [a, b, c] for the line equation
      :math:`ax + by + c = 0` where :math:`\\sqrt{a^2+b^2}=1`.
    - In **3D**, the line is estimated by performing PCA (Principal Component Analysis) on the points to find
      the main direction of variance.

    Arguments:
        points (array-like): A list or array of shape ``(N, 2)`` or ``(N, 3)``, containing at least two points.

    Returns:
        list:
            - In 2D: Normalized line coefficients ``[a, b, c]``.
            - In 3D: A list [point, direction_vector], where both elements are lists of 3 floats.

    Raises:
        AssertionError: If the input does not have 2D or 3D shape.
    '''
    def two_dim():
        A = np.ones((points.shape[0], 2))
        b = np.ones((points.shape[0], 1))
        for i in range(points.shape[0]):
            A[i, 0] = points[i, 0]
            b[i, 0] = points[i, 1]
        x = np.linalg.inv(A.T @ A) @ A.T @ b
        a, b = x.flatten()
        # y = ax + b, or ax - y + b = 0
        return (np.array([a, -1, b]) / norm([a, -1])).tolist()

    def three_dim():
        center = np.mean(points, axis=0)
        pc = points - center
        cov = np.cov(pc, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        max_eigen_index = np.argmax(eigenvalues)
        direction_vec = eigenvectors[:, max_eigen_index]
        return [center.tolist(), direction_vec.tolist()]

    points = np.array(points)
    if points.shape[1] == 2:
        return two_dim()
    elif points.shape[1] == 3:
        return three_dim()

    assert False


def point_from_plane_and_line(plane, line):
    '''Calculate the intersection point of a plane and a line in 3D space.

    Arguments:
        plane (list or array-like): Representation of the plane as ``[a, b, c, d]``
                                    where the plane equation is :math:`ax + by + cz + d = 0`
        line (list or array-like): Representation of the line as ``[[x1, y1, z1], [vx, vy, vz]]``
                                   where ``[x1, y1, z1]`` is a point on the line and
                                   ``[vx, vy, vz]`` is the direction vector of the line

    Returns:
        numpy.ndarray: Coordinates of the intersection point ``[x, y, z]`` if it exists,
                       otherwise a warning message indicating that the line lies on the plane
                       or is parallel to the plane and an array with 3 ``NaN`` is returned

    Raises:
        AssertionError: If the dimensions of the plane and line do not match the expected dimensions
    '''
    assert len(plane) == 4 and np.array(line).shape == (2, 3)
    den = np.dot(plane[:3], line[1])
    if np.isclose(den, 0.0, atol=1e-5):
        if is_line_on_plane(line, plane):
            warnings.warn('The line lies on the plane.')
        else:
            warnings.warn('The line is parallel with the plane but does not intersect it.')
        return np.array([np.nan] * 3)
    t = - np.dot(plane, np.append(line[0], 1.0)) / np.dot(plane[:3], line[1])
    return np.array(line[0]) + t * np.array(line[1])


def point_from_three_planes(plane_1, plane_2, plane_3):
    '''Calculate the intersection point of three planes in 3D space.

    Arguments:
        plane_1 (list or array-like): Representation of the plane as ``[a, b, c, d]``
                                      where the plane equation is :math:`ax + by + cz + d = 0`
        plane_2 (list or array-like): same as `plane_1`
        plane_3 (list or array-like): same as `plane_1`

    Returns:
        numpy.ndarry: Coordinates of the intersection point ``[x, y, z]``

    Raises:
        AssertionError: If the dimensions of the planes do not match the expected dimensions.

    Note:
        An array with 3 ``NaN`` is returned if these planes are not intersect at one point
    '''
    assert len(plane_1) == len(plane_2) == len(plane_3) == 4
    try:
        line = line_from_planes(plane_1, plane_2)
    except AssertionError:
        warnings.warn('plane_1 and plane_2 are coincident or parallel')
        return np.array([np.nan] * 3)
    return point_from_plane_and_line(plane_3, line)


def circle_from_three_points(p1, p2, p3):
    '''Calculate the circle passing through three points in 2D or 3D space.

    Arguments:
        p1 (list or array-like): Coordinates of the first point
        p2 (list or array-like): Coordinates of the second point
        p3 (list or array-like): Coordinates of the third point

    Returns:
        tuple: A tuple containing:

          - numpy.ndarray: Coordinates of the circle's center
          - float: Radius of the circle
          - numpy.ndarray:
            Plane coefficient ``[a, b, c, d]`` which the circle lies in.
            For 2D, this term is neglectable and XY plane (:math:`Z=0`) is returned

    Raises:
        AssertionError: If the points do not lie in the same dimension
        AssertionError: If the dimension of the point is not 2 nor 3
        AssertionError: If three points are collinear
    '''
    def two_dim():
        line_1 = perpendicular_bisector(p1, p2)
        line_2 = perpendicular_bisector(p2, p3)
        center = intersection_between_lines(line_1, line_2)
        radius = distance_between_points(p1, center)
        return center, radius, np.array(XY_PLANE)

    def three_dim():
        plane_1 = perpendicular_bisector(p1, p2)
        plane_2 = perpendicular_bisector(p2, p3)
        plane_3 = plane_from_three_points(p1, p2, p3)
        center = point_from_three_planes(plane_1, plane_2, plane_3)
        radius = distance_between_points(p1, center)
        return center, radius, plane_3

    if len(p1) == 2:
        return two_dim()
    elif len(p2) == 3:
        return three_dim()
    assert False


def circle_from_center_and_points(center, p1, p2=None):
    '''Calculate the circle given its center and two points on its circumference.

    Arguments:
        center (list or array-like): Coordinates of the circle's center
        p1 (list or array-like): Coordinates of the first point on the circumference
        p2 (list or array-like): Coordinates of the second point on the circumference.
                                 Neglectable for 2D case

    Returns:
        tuple:
            A tuple containing

          - numpy.ndarray: Coordinates of the circle's center
          - float: Radius of the circle
          - list:
            Plane coefficient ``[a, b, c, d]`` which the circle lies on.
            For 2D, this term is neglectable and XY plane (:math:`Z=0`) is returned

    Raises:
        AssertionError: If the points do not lie in the same dimension
        AssertionError: If the dimension of the point is not 2 nor 3
        AssertionError: If the distances from the center to p1 and p2 are not equal within a tolerance.
    '''
    def two_dim():
        return center, distance_between_points(center, p1), XY_PLANE

    def three_dim():
        return center, distance_between_points(center, p1), plane_from_three_points(center, p1, p2)

    if len(center) == 2:
        assert len(center) == len(p1)
        return two_dim()
    elif len(center) == 3:
        assert len(center) == len(p1) == len(p2)
        assert np.isclose(distance_between_points(center, p1), distance_between_points(center, p2), atol=1e-5)
        return three_dim()
    assert False


def circle_from_noisy_data(data):
    '''Calculate the circle given by noised data.

    Arguments:
        data (list or array-like): sampled noised data, should with shape either ``(N, 2)`` or ``(N, 3)``.
                                   N should greater than or equal to 4.

    Returns:
        tuple:
          - center (numpy.ndarray): coordinates of the circle's center
          - radius (float): radius of the circle
          - plane (list):
            plane coefficient ``[a, b, c, d]`` which the circle lies on.
            For 2D, this term is neglectable and XY plane is returned

    Raise:
        AssertionError: if N less then 4
        AssertionError: if the dimension of the point is not 2 nor 3

    .. todo::
        3D is not supported yet!
    '''
    def two_dim():
        A = np.ones((len(data), 3))
        b = np.zeros((len(data), 1))
        for i in range(len(data)):
            A[i, 0] = data[i, 0]  # xi
            A[i, 1] = data[i, 1]  # yi
            b[i, 0] = -(data[i, 0]**2 + data[i, 1]**2)
        x, _, _, _ = np.linalg.lstsq(A, b, None)
        xc = -x[0, 0] / 2
        yc = -x[1, 0] / 2
        radius = np.sqrt(xc**2 + yc**2 - x[2, 0])
        return (xc, yc), radius, XY_PLANE

    def three_dim():
        raise NotImplementedError()

    data = np.array(data)
    assert data.shape[0] >= 4, 'Should with at least 4 data points to determine the parameters of the circle'
    if data.shape[1] == 2:
        return two_dim()
    elif data.shape[1] == 3:
        return three_dim()
    assert False


def circle_coordinate_transform(center, plane, x_direction_point=None):
    '''Calculate the transformation matrix from a new coordinate system to the original one. The transformed circle is
    centered at origin and lies in XY plane (:math:`Z=0`).

    Arguments:
        center (list or array-like): Coordinates of the circle's center
        plane (list or array-like): Plane coefficient ``[a, b, c, d]`` which the circle lies in
        x_direction_point (list or array-like or None): Point in the plane to determine the x-axis direction.
                                                        If not provided, a random point on the plane is chosen.

    Returns:
        numpy.ndarray:
            Transformation matrix T of shape (4, 4) that transforms coordinates into the original coordinate system

    Raises:
        AssertionError: If the dimensions of center or plane are incorrect
        AssertionError: If `x_direction_point` is provided but its dimensions are not 3.
    '''
    assert len(center) == 3 and len(plane) == 4
    if x_direction_point:
        assert len(x_direction_point) == 3
    else:
        x_direction_point = random_point_on_plane(plane)
    z_axis = unit_vector(plane[:3])
    x_axis = unit_vector(np.array(x_direction_point) - np.array(center))
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)  # convert position in new coordinate into original coordinate
    T[:3, :3] = np.vstack([x_axis, y_axis, z_axis]).T
    T[:3, 3] = center
    return T


def arc_from_center_and_endpoints(center, p1, p2):
    '''Compute the parameters of an arc defined by its center and endpoints.

    Arguments:
        center (list or array-like): Center coordinates of the arc
        p1 (list or array-like): First endpoint coordinates of the arc
        p2 (list or array-like): Second endpoint coordinates of the arc

    Returns:
        tuple:
            A tuple containing:

            - numpy.ndarray: Center of the arc
            - float: Radius of the arc
            - list: angle range of the arc

                - 2D: angle is relative to X-axis
                - 3D: angle is relative to the direction of `center` to `p1`
            - numpy.ndarray: transformation matrix to convert arc coordinates.

                - 2D: the matrix is with shape 3x3
                - 3D: the matrix is with shape 4x4

    Raises:
        AssertionError: If the dimensions of center, p1, or p2 are not consistent
        AssertionError: If the dimension of the point is not 2 nor 3
        AssertionError: If the distances from the center to p1 and p2 are not equal within a tolerance.
    '''
    def two_dim():
        center_, radius, _ = circle_from_center_and_points(center, p1, p2)
        vec_1 = np.array(p1) - np.array(center)
        vec_2 = np.array(p2) - np.array(center)
        theta_1 = np.arctan2(vec_1[1], vec_1[0])
        theta_2 = np.arctan2(vec_2[1], vec_2[0])
        thetas = [theta_1, theta_2]
        T = np.eye(3)
        T[:2, 2] = center_
        return center, radius, thetas, T

    def three_dim():
        center_, radius, plane = circle_from_center_and_points(center, p1, p2)
        vec_1 = np.array(p1) - np.array(center)
        vec_2 = np.array(p2) - np.array(center)
        T = circle_coordinate_transform(center, plane, p1)
        theta_2 = angle_between_vectors(vec_1, vec_2)
        return center, radius, [0, theta_2], T

    assert len(center) == len(p1) == len(p2)
    assert np.isclose(distance_between_points(center, p1), distance_between_points(center, p2), atol=1e-5)
    if len(center) == 2:
        return two_dim()
    elif len(center) == 3:
        return three_dim()
    assert False


def arc_from_three_points(p1, p2, p3):
    '''Construct an arc passing through three points in 2D or 3D space.

    Arguments:
        p1 (list or array-like): First point on the arc.
        p2 (list or array-like): Second point on the arc.
        p3 (list or array-like): Third point on the arc.

    Returns:
        tuple:
            A tuple containing:

            - numpy.ndarray: Center of the arc
            - float: Radius of the arc
            - list: angle range of the arc

                - 2D: angle is relative to X-axis
                - 3D: angle is relative to the direction of `center` to `p1`
            - numpy.ndarray: transformation matrix to convert arc coordinates.

                - 2D: the matrix is with shape 3x3
                - 3D: the matrix is with shape 4x4

    Raises:
        AssertionError: If input point dimensions are invalid or inconsistent.
    '''
    def two_dim():
        line_1 = perpendicular_bisector(p1, p2)
        line_2 = perpendicular_bisector(p1, p3)
        center = intersection_between_lines(line_1, line_2)
        return arc_from_center_and_endpoints(center, p1, p3)

    def three_dim():
        plane_1 = perpendicular_bisector(p1, p2)
        plane_2 = perpendicular_bisector(p1, p3)
        plane_3 = plane_from_three_points(p1, p2, p3)
        center = point_from_three_planes(plane_1, plane_2, plane_3)
        return arc_from_center_and_endpoints(center, p1, p3)

    if len(p1) == 2:
        return two_dim()
    elif len(p1) == 3:
        return three_dim()
    assert False


def generate_points_on_circle(center, radius, plane, num=50):
    '''Generate `num` points on a circle given its center and radius.

    Arguments:
        center (list or array-like): Coordinates of the circle's center
        radius (float): Radius of the circle
        plane (list or array-like): Plane coefficient ``[a, b, c, d]`` which the circle lies on.
                                    For 2D, this term is neglectable.
        num (int, optional): Number of points to generate on the circle, default is 50

    Returns:
        numpy.ndarray: Array of points lying on the circle

    Raises:
        AssertionError: If the dimensions of center, radius, or plane are incorrect.
        AssertionError: If got negative radius
    '''
    def two_dim():
        points = []
        for theta in np.linspace(0, 2 * np.pi, num):
            p = [
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
            ]
            points.append(p)
        return np.array(points)

    def three_dim():
        points = []
        T = circle_coordinate_transform(center, plane)
        for theta in np.linspace(0, 2 * np.pi, num):
            p = [radius * np.cos(theta), radius * np.sin(theta), 0.0, 1.0]
            points.append((T @ p)[:3])
        return np.array(points)

    assert radius > 0, 'Radius should be positive'
    if len(center) == 2:
        return two_dim()
    elif len(center) == 3:
        assert len(plane) == 4
        return three_dim()
    assert False


def generate_points_on_arc(center, radius, theta_range, transform, num=50):
    '''Generate `num` points on an arc given its center, radius, range of theta and the transform.

    Arguments:
        center (numpy.ndarray): Center of the arc
        radius (float): Radius of the arc
        thetas (list): Angle range of the arc

            - 2D: angle is relative to X-axis
            - 3D: angle is relative to X-axis in new coordinate
        T (numpy.ndarray): Transformation matrix to convert arc coordinates into original coordinate

            - 2D: the matrix is with shape 3x3
            - 3D: the matrix is with shape 4x4
        num (int, optional): Number of points to generate on the circle, default is 50

    Returns:
        numpy.ndarray: Array of points lying on the arc

    Raises:
        AssertionError: If the dimensions of center is incorrect
        AssertionError: If got negative radius
        AssertionError: If the dimension of center is mismatched withe the transform
    '''
    assert len(center) in [2, 3] and len(theta_range) == 2
    assert radius > 0
    assert np.array(transform).shape == (len(center) + 1, len(center) + 1)
    thetas = np.linspace(theta_range[0], theta_range[1], num)
    points = []
    for theta in thetas:
        p = [0.0] * len(center)
        p[0] = radius * np.cos(theta)
        p[1] = radius * np.sin(theta)
        p.append(1.0)
        points.append((transform @ p)[:len(center)])
    return np.array(points)


def intersection_between_line_and_circle(line, circle):
    '''Find intersection points between a line and a circle in 2D or 3D space.

    Arguments:
        line (list or tuple): Line parameters depending on dimensionality:

            - For 2D: ``[a, b, c]`` Representing the line equation :math:`ax + by + c = 0`.
            - For 3D: ``[[x0, y0, z0], [vx, vy, vz]]`` representing a point and direction vector respectively.
        circle (tuple):
            - center (list or array-like): Coordinates of the circle's center
            - radius (float): Radius of the circle
            - plane (list or array-like): Plane coefficient ``[a, b, c, d]`` which the circle lies on.
                                          For 2D, this term is neglectable.

    Returns:
        list: List of intersection point(s)

    Raises:
        AssertionError: If the dimensions of center, radius, or plane are incorrect.
        AssertionError: If got negative radius
    '''
    def two_dim():
        assert len(line) == 3
        center, radius = circle[:2]
        assert len(center) == 2
        distance = distance_point_to_line(center, line)
        if distance > radius:
            return []
        elif np.isclose(distance, radius, atol=1e-5):
            return [project_point_on_line(center, line)]
        else:
            mp = project_point_on_line(center, line)
            direction = unit_vector([line[1], -line[0]])
            distance = np.sqrt(radius * radius - distance * distance)
            p1 = (np.array(mp) + direction * distance).tolist()
            p2 = (np.array(mp) - direction * distance).tolist()
            return [p1, p2]

    def three_dim():
        assert np.array(line).shape == (2, 3)
        assert len(circle) == 3
        center, radius, plane = circle
        assert len(center) == 3 and len(plane) == 4
        if not is_line_on_plane(line, plane):
            p = point_from_plane_and_line(plane, line)
            if np.isclose(distance_between_points(p, center), radius, atol=1e-5):
                return [p]
        T = circle_coordinate_transform(center, plane, np.array(center) + np.array(line[1]))
        T_ = np.linalg.inv(T)  # convert position in original coordinate into new coordinate
        p_new = (T_ @ np.append(line[0], 1))[:2]  # remove z
        circle_new = [[0.0, 0.0], radius, XY_PLANE]
        line_new = line_from_point_vector(p_new, [1.0, 0.0])
        intersection_points_new = intersection_between_line_and_circle(line_new, circle_new)
        intersection_points_origin = []
        for p in intersection_points_new:
            p_ = np.append(p, [0.0, 1.0])
            intersection_points_origin.append((T @ p_).tolist()[:3])
        return intersection_points_origin

    assert len(circle) in [2, 3]
    assert circle[1] > 0, 'Radius should be positive'
    if len(line) == 3:
        return two_dim()
    elif len(line) == 2:
        return three_dim()
    assert False


def point_circle_relation(point, circle):
    '''Get relation between a point and a circle.

    Arguments:
        point (list or array-like): Point coordinates to check.
        circle (tuple):
            - center (list or array-like): Coordinates of the circle's center
            - radius (float): Radius of the circle
            - plane (list or array-like):
                Plane coefficient ``[a, b, c, d]`` which the circle lies on.
                For 2D, this term is neglectable.

    Returns:
        bool: True if the point lies on the circle, False otherwise
        PointCircleEnum: Enumeration as following:

        - 0: If the point is inside the circle
        - 1: If the point is on the border of the circle
        - 2: If the point is outside the circle
        - 3: If the point is not on the plane of the circle (3D only)

    Raises:
        AssertionError: if the dimensions of center, radius, or plane (only in 3D) are incorrect.
        AssertionError: if got negative radius
    '''
    def _compare_distance(point, center, radius):
        dist = distance_between_points(center, point)
        if np.isclose(dist, radius, atol=1e-5):
            return PointCircleEnum.ON_BORDER
        elif dist < radius:
            return PointCircleEnum.INSIDE
        return PointCircleEnum.OUTSIDE

    def two_dim():
        assert len(circle) >= 2
        center, radius = circle[:2]
        assert len(center) == 2
        return _compare_distance(point, center, radius)

    def three_dim():
        assert len(circle) == 3
        center, radius, plane = circle
        assert len(center) == 3 and len(plane) == 4 and radius > 0
        if not is_point_on_plane(point, plane):
            return PointCircleEnum.NOT_IN_PLANE
        return _compare_distance(point, center, radius)

    if len(point) == 2:
        return two_dim()
    elif len(point) == 3:
        return three_dim()
    assert False


def is_point_on_arc(point, arc):
    '''Determines if a point lies on a given arc.

    Arguments:
        point (list or array-like): Coordinates of the point to check
        arc (tuple): A tuple contains center, radius, theta_range, and transform.

            - center (numpy.ndarray): Center of the arc
            - radius (float): Radius of the arc
            - thetas (list): Angle range of the arc

                - 2D: angle is relative to X-axis
                - 3D: angle is relative to the direction of `center` to `p1`
            - T (numpy.ndarray): Transformation matrix to convert arc coordinates.

                - 2D: the matrix is with shape 3x3
                - 3D: the matrix is with shape 4x4

    Returns:
        bool: True if the point is on the arc, False otherwise

    Raises:
        AssertionError: If the dimensions of the inputs are not consistent.
        AssertionError: If the length of arc is incorrect
        AssertionError: If the radius is negative

    .. todo::
        3D is not supported yet!
    '''
    def two_dim():
        theta = angle_between_vectors([1, 0], np.array(point) - np.array(center))
        if theta_range[0] <= theta <= theta_range[1]:
            return True
        return False

    def three_dim():
        # TODO
        pass

    assert len(arc) == 4, 'Arc should contain center, radius, theta_range and transform'
    center, radius, theta_range, T = arc
    assert len(point) == len(center), 'Dimension of point and center is not consistent'
    shape = (len(point) + 1, len(point) + 1)
    assert np.array(T).shape == shape, \
           f'Shape of transform expected to be {shape}, got {np.array(T).shape}'
    assert radius > 0, 'Radius should be positive'

    if not np.isclose(distance_between_points(point, center), radius, atol=1e-5):
        return False
    if len(point) == 2:
        return two_dim()
    elif len(point) == 3:
        return three_dim()
    assert False


def point_triangle_relation(p, p1, p2, p3):
    '''Determine the relation between `p` and the triangle formed by `p1`, `p2` and `p3`.

    Arguments:
        p (list or array-like): Coordinates of the point to check
        p1 (list or array-like): Coordinates of the first vertex of the triangle
        p2 (list or array-like): Coordinates of the second vertex of the triangle
        p3 (list or array-like): Coordinates of the third vertex of the triangle

    Return:
        PointTriangleEnum: Enumeration as following

        - 0: If point is inside the triangle
        - 1: If point in on the border of the triangle
        - 2: If point is outside the triangle
        - 3: If point is not in the plane formed by three vertices (3D only)

    Raise:
        AssertionError: If the dimensions of the points are not consistent.
    '''
    assert len(p) == len(p1) == len(p2) == len(p3), 'Dimension of points are not consistent'
    if len(p) == 3 and not is_line_on_plane(p, plane_from_three_points(p1, p2, p3)):
        return PointTriangleEnum.NOT_IN_PLANE
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    v = np.array(p) - np.array(p1)
    # v = s * v1 + t * v2
    s = np.dot(v1, v) / norm(v1) ** 2
    t = np.dot(v2, v) / norm(v2) ** 2
    if s > 0 and t > 0 and s + t < 1:
        return PointTriangleEnum.INSIDE
    elif s == 0 or t == 0 or s + t == 1:
        return PointTriangleEnum.ON_BORDER
    return PointTriangleEnum.OUTSIDE


def point_cylinder_relation(p, cylinder):
    '''Determine the spatial relationship between a point and a cylinder.

    Arguments:
        p (list or array-like): Coordinates of the 3D point to be tested.
        cylinder (tuple): A tuple (bottom_center, radius, height, direction), where:

            - bottom_center (list or array-like): The 3D coordinates of the cylinder's base center.
            - radius (float): Radius of the cylinder (must be positive).
            - height (float): Height of the cylinder (must be positive).
            - direction (list or array-like): A 3D vector indicating the axis direction.

    Returns:
        PointShapeEnum: Enumeratiion as following:

        - 0: If point is inside the cylinder
        - 1: If point is on the border of the cylinder
        - 2: If point is out of the cylinder

    Raises:
        AssertionError: If the cylinder parameters are invalid or improperly formatted.
    '''
    assert len(cylinder) == 4, 'Invalid cylinder'
    assert len(cylinder[0]) == len(cylinder[3]) == 3, 'First and fourth data should be with length 3'
    assert cylinder[1] > 0, 'Radius should be positive'
    assert cylinder[2] > 0, 'Height should be positive'
    bottom_center, radius, height, direction = cylinder
    direction = unit_vector(direction)
    ov = orthogonal_vector(direction)
    ov2 = np.cross(direction, ov)
    axis_line = line_from_point_vector(cylinder[0], cylinder[3])
    bottom_circle_point1 = np.array(bottom_center) + ov * radius
    bottom_circle_point2 = np.array(bottom_center) + ov2 * radius
    bottom_circle = circle_from_center_and_points(bottom_center, bottom_circle_point1, bottom_circle_point2)
    top_center = np.array(bottom_center) + direction * height
    top_circle_point1 = np.array(top_center) + ov * radius
    top_circle_point2 = np.array(top_center) + ov2 * radius
    top_circle = circle_from_center_and_points(top_center, top_circle_point1, top_circle_point2)
    if (point_circle_relation(p, bottom_circle) in [PointCircleEnum.INSIDE, PointCircleEnum.ON_BORDER] or
            point_circle_relation(p, top_circle) in [PointCircleEnum.INSIDE, PointCircleEnum.ON_BORDER]):
        return PointShapeEnum.ON_BORDER
    if np.isclose(
            distance_point_to_plane(p, bottom_circle[2]) + distance_point_to_plane(p, top_circle[2]),
            height,
            atol=1e-5):
        dist = distance_point_to_line(p, axis_line)
        if np.isclose(dist, radius, atol=1e-5):
            return PointShapeEnum.ON_BORDER
        elif dist < radius:
            return PointShapeEnum.INSIDE
        return PointShapeEnum.OUTSIDE
    return PointShapeEnum.OUTSIDE


def spherical_cap_volume(radius, height):
    '''Calculate the volume of a spherical cap.

    A spherical cap is the portion of a sphere cut off by a plane. The formula used is:
    :math:`V = \\frac{\\pi h^2 (3R - h)}{3}`

    Arguments:
        radius (float): Radius of the sphere (must be positive).
        height (float): Height of the cap (must be in [0, 2 * radius)).

    Returns:
        float: Volume of the spherical cap.

    Raises:
        AssertionError: If the radius is not positive, height is negative, or exceeds the sphere's diameter.
    '''
    assert radius > 0, "Radius should be non-negative"
    assert height >= 0, "Height should be greater than or equal to 0"
    assert height < 2 * radius, "Height should not greater than two times of radius"
    return np.pi * height ** 2 * (3 * radius - height) / 3


def overlap_volume_between_spheres(sphere1, sphere2):
    '''Compute the volume of intersection (overlap) between two spheres.

    The overlap volume is calculated using the analytical formula for intersecting spheres
    based on the distance between centers and their respective radii.

    Arguments:
        sphere1 (tuple): A tuple (center, radius), where:

            - center (list or array-like): Coordinates of the first sphere center.
            - radius (float): Radius of the first sphere (must be positive).
        sphere2 (tuple): Same format as `sphere1`, representing the second sphere.

    Returns:
        float: Volume of the overlapping region. Returns 0 if spheres do not intersect.

    Raises:
        AssertionError: If sphere definitions are invalid or contain non-positive radii.
    '''
    assert len(sphere1) == 2 and len(sphere2) == 2, "Invalid sphere input"
    assert len(sphere1[0]) == 3 and len(sphere2[0]) == 3, "Invalid sphere input"
    assert sphere1[1] > 0 and sphere2[1] > 0, "Invalid sphere input"
    d = distance_between_points(sphere1[0], sphere2[0])
    r1, r2 = sphere1[1], sphere2[1]
    if d >= r1 + r2:
        return 0
    return np.pi * (r1 + r2 - d)**2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2)**2) / (12 * d)


def random_point_on_sphere(sphere):
    '''Generate a random point uniformly distributed on the surface of a sphere.

    Arguments:
        sphere (tuple): A tuple (center, radius), where:

            - center (list or array-like): Coordinates of the sphere's center (3D).
            - radius (float): Radius of the sphere (must be positive).

    Returns:
        numpy.ndarray: A 3D coordinate of the random point on the sphere surface.

    Raises:
        AssertionError: If the input is not a valid sphere definition.
    '''
    assert len(sphere) == 2 and sphere[1] > 0 and len(sphere[0]) == 3
    center, radius = sphere
    theta = np.random.random() * 2 * np.pi  # azimuth
    u = np.random.random()
    phi = np.arccos(1 - 2 * u)  # zenith
    return np.array(center) + spherical_to_cartesian((radius, theta, phi))


def point_sphere_relation(point, sphere):
    '''Determine the spatial relationship between a point and a sphere.

    Arguments:
        point (list or array-like): 3D coordinates of the point to be tested.
        sphere (tuple): A tuple (center, radius), where:

            - center (list or array-like): 3D coordinates of the sphere's center.
            - radius (float): Radius of the sphere (must be positive).

    Returns:
        PointShapeEnum: Enumeration as following

            - 0: If point is inside the cylinder
            - 1: If point is on the border of the cylinder
            - 2: If point is out of the cylinder
    Raises:
        AssertionError: If the input point or sphere is invalid or improperly formatted.
    '''
    assert len(sphere) == 2 and sphere[1] > 0 and len(sphere[0]) == 3 and len(point) == 3
    center, radius = sphere
    dist = distance_between_points(point, center)
    if np.isclose(dist, radius, atol=1e-5):
        return PointShapeEnum.ON_BORDER
    elif dist < radius:
        return PointShapeEnum.INSIDE
    return PointShapeEnum.OUTSIDE


def cartesian_to_spherical(point):
    '''
    Convert a 3D point from Cartesian coordinates to spherical coordinates.

    Arguments:
        point (list or array-like): A 3D point ``[x, y, z]``.

    Returns:
        numpy.ndarray: A 3-element array ``[radius, theta, phi]`` where

        - radius: Distance from the origin to the point (must be non-zero).
        - theta (azimuth): Angle in the XY plane from the X-axis, in radians :math:`[0, 2\\pi)`.
        - phi (zenith): Angle from the positive Z-axis, in radians :math:`[0, \\pi]`.

    Raises:
        AssertionError: If the input point is not 3D or the radius is zero.
    '''
    assert len(point) == 3
    x, y, z = point
    radius = norm(point)
    assert not np.isclose(radius, 0.0, atol=1e-5)
    if np.isclose(x, 0.0, atol=1e-5) and np.isclose(y, 0.0, atol=1e-5):
        theta = 0.0
        if z > 0:
            phi = 0.0
        else:
            phi = np.pi
    else:
        theta = np.arctan2(y, x)
        phi = np.arccos(z / radius)
    return np.array([radius, theta, phi])


def spherical_to_cartesian(point):
    '''Convert a 3D point from spherical coordinates to Cartesian coordinates.

    Arguments:
        point (list or array-like): A 3-element list or array ``[radius, theta, phi]``.

    Returns:
        numpy.ndarray: A 3D point ``[x, y, z]`` in Cartesian coordinates.

    Raises:
        AssertionError: If the input is not valid spherical coordinates.
    '''
    assert len(point) == 3
    radius, azimuth, zenith = point
    assert radius >= 0 and 0 <= azimuth < 2 * np.pi and 0 <= zenith <= np.pi
    return np.array([
        radius * np.cos(azimuth) * np.sin(zenith),
        radius * np.sin(azimuth) * np.sin(zenith),
        radius * np.cos(zenith)
    ])


def distance_between_points_on_sphere(p1, p2, sphere):
    '''Calculate the shortest distance (i.e., geodesic) between two points on the surface of a sphere.

    The function assumes that both points lie exactly on the surface of the sphere.
    The distance is computed as the product of the sphere's radius and the angle (in radians)
    between the vectors from the sphere center to each point.

    Arguments:
        p1 (list or array-like): Coordinates of the first point on the sphere surface.
        p2 (list or array-like): Coordinates of the second point on the sphere surface.
        sphere (tuple): A tuple (center, radius), where:

            - center (list or array-like): Coordinates of the sphere's center.
            - radius (float): Radius of the sphere (must be positive).

    Returns:
        float: The great-circle distance between the two points along the sphere's surface.

    Raises:
        AssertionError: If inputs are not valid 3D points
        AssertionError: If radius is non-positive
        AssertionError: If points are not on the sphere surface
    '''
    assert len(p1) == len(p2) == 3
    assert len(sphere) == 2 and sphere[1] > 0 and len(sphere[0]) == 3
    assert point_sphere_relation(p1, sphere) == PointShapeEnum.ON_BORDER
    assert point_sphere_relation(p2, sphere) == PointShapeEnum.ON_BORDER
    center, radius = sphere
    v1 = np.array(p1) - np.array(center)
    v2 = np.array(p2) - np.array(center)
    angle = angle_between_vectors(v1, v2)
    return radius * angle
