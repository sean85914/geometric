import numpy as np


XY_PLANE = [0.0, 0.0, 1.0, 0.0]
YZ_PLANE = [1.0, 0.0, 0.0, 0.0]
ZX_PLANE = [0.0, 1.0, 0.0, 0.0]


def norm(vector):
    return np.linalg.norm(vector)


def is_zero_vector(vector):
    return np.isclose(norm(vector), 0.0, atol=1e-5)


def unit_vector(vector):
    assert not is_zero_vector(vector)
    return vector / np.linalg.norm(vector)


def is_on_axis(vector):
    unit = unit_vector(vector)
    for idx, component in enumerate(unit):
        if np.isclose(abs(component), 1.0, atol=1e-5):
            return True, idx
    return False, -1


def middle_point(p1, p2):
    assert len(p1) == len(p2)
    return ((np.array(p1) + np.array(p2)) / 2).tolist()


def average_point(*p, **kwargs):
    p = np.array(p)
    return np.average(p, weights=kwargs.get('weights', None), axis=0)


def distance_between_points(p1, p2):
    assert len(p1) == len(p2)
    return np.linalg.norm(np.array(p1) - np.array(p2))


def line_from_point_vector(point, vector):
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
    assert False


def line_from_two_points(p1, p2):
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
    assert False


def plane_from_three_points(p1, p2, p3):
    assert len(p1) == len(p2) == len(p3) == 3
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    assert not np.isclose(np.linalg.norm(normal), 0, atol=1e-5)
    normal = unit_vector(normal)
    const = np.dot(normal, p1)
    nx, ny, nz = normal
    return [nx, ny, nz, -const]


def plane_from_point_vector(point, vector):
    assert len(point) == len(vector) == 3
    vector = unit_vector(vector)
    return np.array(np.append(vector, -np.dot(vector, point))).tolist()


def perpendicular_bisector(p1, p2):
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
    assert False


def angle_bisector(line_1, line_2):
    pass
    # TODO


def nearest_point(point, vector, target_point):
    '''Given a point and a vector, as well as another point, return the
    nearest point on the line passing through the point in the direction
    of the vector and the other point.

    Arguments:
        point (list or array-like): coordinates of the point through which the line passes
        vector (list or array-like): direction vector of the line
        target_point (list or array-like): coordinates of the target point

    Return:
        nearest_point (numpy.array): the nearest point on the line
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
        point (list or array-like): coordinates of the point through which the line passes
        vector (list or array-like): direction vector of the line
        target_point (list or array-like): coordinates of the target point

    Return:
        dist (float): the shortest distance between the line and the target point
    '''
    n_point = nearest_point(point, vector, target_point)
    return np.linalg.norm(n_point - np.array(target_point))


def angle_between_vectors(v1, v2, degrees=False):
    assert len(v1) == len(v2)
    assert not (is_zero_vector(v1) and is_zero_vector(v2))
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(min(max(c, -1), 1))
    if degrees:
        return np.degrees(theta)
    return theta


def vector_projection(v1, v2):
    '''Project a vector, v1, onto another vector, v2.

    Arguments:
        v1 (list or array-like): vector to be projected
        v2 (list or array-like):

    Return:
        v_project (numpy.array):
    '''
    assert len(v1) == len(v2)
    assert not is_zero_vector(v2)
    norm_square = np.power(np.linalg.norm(v2), 2)
    dot = np.dot(v1, v2)
    return dot / norm_square * np.array(v2)


def project_vector_on_plane(v, normal):
    '''Project a vector, v, onto a plane represented by its normal vector.

    Arguments:
        v (list or array-like):
        normal (list or array-like): normal vector of the plane

    Return:
        v_project (numpy.array):
    '''
    assert len(v) == len(normal) == 3
    v_parallel = vector_projection(v, normal)
    return v - v_parallel


def point_on_line(point, line):
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


def point_on_plane(point, plane):
    assert len(plane) == 4 and len(point) == 3
    x, y, z = point
    return np.isclose(np.dot(plane, [x, y, z, 1.0]), 0, atol=1e-5)


def line_on_plane(line, plane):
    assert np.array(line).shape == (2, 3) and len(plane) == 4
    point, vector = line
    if np.isclose(np.dot(vector, plane[:3]), 0.0, atol=1e-5) and point_on_plane(point, plane):
        return True
    return False


def project_point_on_line(p, line):
    assert len(p) == 2 and len(line) == 3
    x, y = p
    a, b, c = line
    norm_square = np.power(np.linalg.norm([a, b]), 2)
    d = -b * x + a * y
    x_p = (-b * d - a * c) / norm_square
    y_p = (a * d - b * c) / norm_square
    return np.array([x_p, y_p])


def project_point_on_plane(p, plane):
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
    assert np.array(line).shape == (2, 3) and len(plane) == 4
    point, vector = line
    pp = project_point_on_plane(point, plane)
    pv = project_vector_on_plane(vector, plane[:3])
    return [pp, pv]


def distance_point_to_line(p, line):
    p_project = project_point_on_line(p, line)
    return np.linalg.norm(p_project - np.array(p))


def distance_point_to_plane(p, plane):
    p_project = project_point_on_plane(p, plane)
    return np.linalg.norm(p_project - np.array(p))


def intersection_between_lines(line_1, line_2):
    def two_dim():
        assert len(line_1) == len(line_2) == 3
        a1, b1, c1 = line_1
        a2, b2, c2 = line_2
        den = np.linalg.det([[a1, b1], [a2, b2]])
        assert not np.isclose(den, 0, atol=1e-5)
        num_x = np.linalg.det([[-c1, b1], [-c2, b2]])
        num_y = np.linalg.det([[a1, -c1], [a2, -c2]])
        return [num_x / den, num_y / den]

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

    if len(line_1) == 3:
        return two_dim()
    elif len(line_1) == 2:
        return three_dim()
    assert False


def intersection_between_line_segments(line_1_points, line_2_points):
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
        return [float('nan')] * dim
    except AssertionError:
        return [float('nan')] * dim


def line_from_planes(plane_1, plane_2):
    assert len(plane_1) == 4 and len(plane_2) == 4
    normal_1 = np.array(plane_1)[:3]
    normal_2 = np.array(plane_2)[:3]
    vector = np.cross(normal_1, normal_2)
    assert not np.isclose(np.linalg.norm(vector), 0.0, atol=1e-5)
    vector = unit_vector(vector)
    if is_on_axis(normal_1)[0] and is_on_axis(normal_2)[0]:
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


def point_from_plane_and_line(plane, line):
    assert len(plane) == 4 and np.array(line).shape == (2, 3)
    t = - np.dot(plane, np.append(line[0], 1.0)) / np.dot(plane[:3], line[1])
    return (np.array(line[0]) + t * np.array(line[1])).tolist()


def point_from_three_planes(plane_1, plane_2, plane_3):
    line = line_from_planes(plane_1, plane_2)
    return point_from_plane_and_line(plane_3, line)


def circle_from_three_points(p1, p2, p3):
    def two_dim():
        line_1 = perpendicular_bisector(p1, p2)
        line_2 = perpendicular_bisector(p2, p3)
        center = intersection_between_lines(line_1, line_2)
        radius = distance_between_points(p1, center)
        return center, radius, XY_PLANE

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


def circle_from_center_and_points(center, p1, p2):
    def two_dim():
        return center, distance_between_points(center, p1), XY_PLANE

    def three_dim():
        return center, distance_between_points(center, p1), plane_from_three_points(center, p1, p2)

    assert len(center) == len(p1) == len(p2)
    assert np.isclose(distance_between_points(center, p1), distance_between_points(center, p2), atol=1e-5)
    if len(center) == 2:
        return two_dim()
    elif len(center) == 3:
        return three_dim()
    assert False


def arc_from_center_and_endpoints(center, p1, p2):
    def two_dim():
        center_, radius, _ = circle_from_center_and_points(center, p1, p2)
        vec_1 = np.array(p1) - np.array(center)
        vec_2 = np.array(p2) - np.array(center)
        theta_1 = angle_between_vectors(vec_1, [1, 0])
        theta_2 = angle_between_vectors(vec_2, [1, 0])
        thetas = [theta_1, theta_2]
        thetas.sort()
        T = np.eye(3)
        T[:2, 2] = center_
        return center, radius, thetas, T

    def three_dim():
        center_, radius, plane = circle_from_center_and_points(center, p1, p2)
        vec_1 = np.array(p1) - np.array(center)
        vec_2 = np.array(p2) - np.array(center)
        z_axis = unit_vector(plane[:3])
        x_axis = unit_vector(vec_1)
        y_axis = np.cross(z_axis, x_axis)
        T = np.eye(4)  # convert position in new coordinate into original coordinate
        T[:3, :3] = np.vstack([x_axis, y_axis, z_axis]).T
        T[:3, 3] = center_
        theta_2 = angle_between_vectors(vec_1, vec_2)
        return center, radius, [0, theta_2], T

    assert len(center) == len(p1) == len(p2)
    assert np.isclose(distance_between_points(center, p1), distance_between_points(center, p2), atol=1e-5)
    if len(center) == 2:
        return two_dim()
    elif len(center) == 3:
        return three_dim()
    assert False


def generate_points_on_circle(center, radius, transform, num):
    pass
    # TODO


def generate_points_on_arc(center, radius, theta_range, transform, num=50):
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
        if not line_on_plane(line, plane):
            p = point_from_plane_and_line(plane, line)
            if np.isclose(distance_between_points(p, center), radius, atol=1e-5):
                return [p]
        z_axis = unit_vector(plane[:3])
        x_axis = unit_vector(line[1])
        y_axis = np.cross(z_axis, x_axis)
        origin = center
        T = np.eye(4)  # convert position in new coordinate into original coordinate
        T[:3, :3] = np.vstack([x_axis, y_axis, z_axis]).T
        T[:3, 3] = origin
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
    if len(line) == 3:
        return two_dim()
    elif len(line) == 2:
        return three_dim()
    assert False


def point_on_circle(point, circle):
    def two_dim():
        assert len(circle) >= 2
        center, radius = circle[:2]
        assert len(center) == 2
        return np.isclose(distance_between_points(center, point), radius, atol=1e-5)

    def three_dim():
        assert len(circle) >= 2
        center, radius = circle[:2]
        assert len(center) == 3
        return (np.isclose(distance_between_points(center, point), radius, atol=1e-5) and
                point_on_plane(point, circle[2]))

    if len(point) == 2:
        return two_dim()
    elif len(point) == 3:
        return three_dim()
    assert False
