import pytest
import numpy as np
from geometric.geometric import (
    norm, is_zero_vector, unit_vector, is_on_axis,
    middle_point, average_point, distance_between_points,
    line_from_point_vector, line_from_two_points,
    random_point_on_line, is_point_on_line,
    plane_from_three_points, plane_from_point_vector, plane_from_noisy_data,
    random_point_on_plane,
    perpendicular_bisector,
    angle_bisector_line_from_two_lines, angle_bisector_plane_from_two_lines,
    angle_bisector_plane_from_two_planes,
    nearest_point, nearest_distance,
    angle_between_vectors, orthogonal_vector, vector_projection, vector_rotation,
    project_vector_on_plane,
    is_point_on_plane, is_line_on_plane,
    project_point_on_line, project_point_on_plane, project_line_on_plane,
    distance_point_to_line, distance_point_to_plane,
    intersection_between_lines, intersection_between_line_segments,
    line_from_planes, line_from_noisy_data,
    point_from_plane_and_line, point_from_three_planes,
    circle_from_three_points, circle_from_center_and_points, circle_from_center_and_radius,
    circle_from_noisy_data, circle_coordinate_transform,
    arc_from_center_and_endpoints, arc_from_three_points,
    generate_points_on_circle, generate_points_on_arc,
    intersection_between_line_and_circle,
    point_circle_relation, is_point_on_arc,
    point_triangle_relation, point_cylinder_relation,
    spherical_cap_volume, overlap_volume_between_spheres,
    random_point_on_sphere, point_sphere_relation,
    cartesian_to_spherical, spherical_to_cartesian,
    distance_between_points_on_sphere,
    parabolic_length, inv_parabolic_length,
    triangle_area, polygon_area,
)
from geometric.relation_enum import PointCircleEnum, PointShapeEnum, PointTriangleEnum


# ── Basic vector utilities ────────────────────────────────────────────────────

class TestNorm:
    def test_zero_vector(self):
        assert norm([0, 0, 0]) == 0.0

    def test_unit_x(self):
        assert np.isclose(norm([1, 0, 0]), 1.0)

    def test_3d(self):
        assert np.isclose(norm([3, 4, 0]), 5.0)

    def test_2d(self):
        assert np.isclose(norm([3, 4]), 5.0)


class TestIsZeroVector:
    def test_zero(self):
        assert is_zero_vector([0, 0, 0])

    def test_nonzero(self):
        assert not is_zero_vector([1, 0, 0])

    def test_near_zero(self):
        assert is_zero_vector([1e-7, 1e-7, 1e-7])


class TestUnitVector:
    def test_x_axis(self):
        u = unit_vector([5, 0, 0])
        assert np.allclose(u, [1, 0, 0])

    def test_arbitrary(self):
        v = [1, 2, 3]
        u = unit_vector(v)
        assert np.isclose(norm(u), 1.0)

    def test_zero_raises(self):
        with pytest.raises(AssertionError):
            unit_vector([0, 0, 0])


class TestIsOnAxis:
    def test_x_axis(self):
        on, idx = is_on_axis([3, 0, 0])
        assert on and idx == 0

    def test_neg_y_axis(self):
        on, idx = is_on_axis([0, -2, 0])
        assert on and idx == 1

    def test_z_axis(self):
        on, idx = is_on_axis([0, 0, 7])
        assert on and idx == 2

    def test_diagonal_not_on_axis(self):
        on, idx = is_on_axis([1, 1, 0])
        assert not on and idx == -1


# ── Point utilities ───────────────────────────────────────────────────────────

class TestMiddlePoint:
    def test_2d(self):
        mp = middle_point([0, 0], [2, 2])
        assert np.allclose(mp, [1, 1])

    def test_3d(self):
        mp = middle_point([0, 0, 0], [4, 0, 0])
        assert np.allclose(mp, [2, 0, 0])

    def test_mismatched_dims_raises(self):
        with pytest.raises(AssertionError):
            middle_point([0, 0], [0, 0, 0])


class TestAveragePoint:
    def test_equal_weight(self):
        avg = average_point([0, 0], [2, 0], [1, 3])
        assert np.allclose(avg, [1, 1])

    def test_weighted(self):
        avg = average_point([0, 0], [2, 0], weights=[1, 1])
        assert np.allclose(avg, [1, 0])


class TestDistanceBetweenPoints:
    def test_same_point(self):
        assert distance_between_points([1, 2, 3], [1, 2, 3]) == 0.0

    def test_unit_distance(self):
        assert np.isclose(distance_between_points([0, 0, 0], [1, 0, 0]), 1.0)

    def test_3_4_5(self):
        assert np.isclose(distance_between_points([0, 0], [3, 4]), 5.0)

    def test_mismatched_dims_raises(self):
        with pytest.raises(AssertionError):
            distance_between_points([0, 0], [0, 0, 0])


# ── Line utilities ────────────────────────────────────────────────────────────

class TestLineFromPointVector:
    def test_2d_point_on_line(self):
        line = line_from_point_vector([0, 0], [1, 0])
        # y=0 line: ax + by + c = 0 → y = 0
        x, y = 3.0, 0.0
        assert np.isclose(line[0] * x + line[1] * y + line[2], 0.0)

    def test_3d_unit_direction(self):
        line = line_from_point_vector([1, 2, 3], [0, 0, 1])
        assert np.allclose(line[0], [1, 2, 3])
        assert np.isclose(norm(line[1]), 1.0)

    def test_invalid_dims_raises(self):
        with pytest.raises(AssertionError):
            line_from_point_vector([0, 0, 0, 0], [1, 0, 0, 0])


class TestLineFromTwoPoints:
    def test_3d_direction(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        assert np.allclose(np.abs(line[1]), [1, 0, 0])

    def test_2d_point_satisfies(self):
        line = line_from_two_points([0, 0], [1, 1])
        for pt in [[0, 0], [1, 1], [0.5, 0.5]]:
            val = line[0] * pt[0] + line[1] * pt[1] + line[2]
            assert np.isclose(val, 0.0, atol=1e-6)


class TestRandomPointOnLine:
    def test_3d_point_on_line(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        for _ in range(10):
            pt = random_point_on_line(line)
            assert is_point_on_line(pt, line)

    def test_2d_point_on_line(self):
        line = line_from_two_points([0, 0], [1, 1])
        for _ in range(10):
            pt = random_point_on_line(line)
            assert is_point_on_line(pt, line)


class TestIsPointOnLine:
    def test_3d_on_line(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        assert is_point_on_line([5, 0, 0], line)

    def test_3d_not_on_line(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        assert not is_point_on_line([0, 1, 0], line)

    def test_2d_on_line(self):
        line = line_from_two_points([0, 0], [1, 1])
        assert is_point_on_line([2, 2], line)

    def test_2d_not_on_line(self):
        line = line_from_two_points([0, 0], [1, 1])
        assert not is_point_on_line([1, 0], line)


# ── Plane utilities ───────────────────────────────────────────────────────────

class TestPlaneFromThreePoints:
    def test_xy_plane(self):
        plane = plane_from_three_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
        # normal should be z-axis
        assert np.isclose(abs(plane[2]), 1.0)
        assert np.isclose(plane[0], 0.0)
        assert np.isclose(plane[1], 0.0)

    def test_point_satisfies_plane(self):
        p1, p2, p3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        plane = plane_from_three_points(p1, p2, p3)
        for pt in [p1, p2, p3]:
            val = np.dot(plane[:3], pt) + plane[3]
            assert np.isclose(val, 0.0, atol=1e-6)

    def test_collinear_raises(self):
        with pytest.raises(AssertionError):
            plane_from_three_points([0, 0, 0], [1, 0, 0], [2, 0, 0])


class TestPlaneFromPointVector:
    def test_xy_plane(self):
        plane = plane_from_point_vector([0, 0, 0], [0, 0, 1])
        assert np.allclose(plane[:3], [0, 0, 1])
        assert np.isclose(plane[3], 0.0)

    def test_offset_plane(self):
        plane = plane_from_point_vector([0, 0, 5], [0, 0, 1])
        # 0*x + 0*y + 1*z + d = 0, z=5 → d=-5
        assert np.isclose(plane[3], -5.0)


class TestPlaneFromNoisyData:
    def test_xy_plane_recovery(self):
        rng = np.random.default_rng(42)
        pts = np.column_stack([
            rng.uniform(-1, 1, 20),
            rng.uniform(-1, 1, 20),
            rng.normal(0, 0.01, 20),
        ])
        plane = plane_from_noisy_data(pts)
        assert np.isclose(abs(plane[2]), 1.0, atol=0.05)


class TestRandomPointOnPlane:
    def test_point_on_plane(self):
        plane = [0, 0, 1, -3]  # z = 3
        for _ in range(10):
            pt = random_point_on_plane(plane)
            val = np.dot(plane[:3], pt) + plane[3]
            assert np.isclose(val, 0.0, atol=1e-5)

    def test_tilted_plane(self):
        plane = plane_from_point_vector([1, 1, 1], [1, 1, 1])
        for _ in range(10):
            pt = random_point_on_plane(plane)
            val = np.dot(plane[:3], pt) + plane[3]
            assert np.isclose(val, 0.0, atol=1e-5)


# ── Perpendicular bisector ────────────────────────────────────────────────────

class TestPerpendicularBisector:
    def test_2d_midpoint_on_bisector(self):
        p1, p2 = [0, 0], [4, 0]
        bisector = perpendicular_bisector(p1, p2)
        mp = middle_point(p1, p2)
        val = bisector[0] * mp[0] + bisector[1] * mp[1] + bisector[2]
        assert np.isclose(val, 0.0, atol=1e-6)

    def test_3d_midpoint_on_bisector_plane(self):
        p1, p2 = [0, 0, 0], [0, 0, 4]
        plane = perpendicular_bisector(p1, p2)
        mp = middle_point(p1, p2)
        val = np.dot(plane[:3], mp) + plane[3]
        assert np.isclose(val, 0.0, atol=1e-6)


# ── Nearest point / distance ─────────────────────────────────────────────────

class TestNearestPoint:
    def test_foot_of_perpendicular(self):
        pt = nearest_point([0, 0, 0], [1, 0, 0], [2, 3, 0])
        assert np.allclose(pt, [2, 0, 0])

    def test_point_on_line_returns_itself(self):
        pt = nearest_point([0, 0, 0], [1, 0, 0], [5, 0, 0])
        assert np.allclose(pt, [5, 0, 0])


class TestNearestDistance:
    def test_perpendicular_distance(self):
        d = nearest_distance([0, 0, 0], [1, 0, 0], [2, 3, 0])
        assert np.isclose(d, 3.0)

    def test_zero_distance_on_line(self):
        d = nearest_distance([0, 0, 0], [1, 0, 0], [5, 0, 0])
        assert np.isclose(d, 0.0)


# ── Orthogonal / projection vector utilities ──────────────────────────────────

class TestOrthogonalVector:
    def test_2d_is_orthogonal(self):
        v = [3, 4]
        u = orthogonal_vector(v)
        assert np.isclose(np.dot(v, u), 0.0)
        assert np.isclose(norm(u), 1.0)

    def test_3d_is_orthogonal(self):
        v = [1, 2, 3]
        u = orthogonal_vector(v)
        assert np.isclose(np.dot(v, u), 0.0, atol=1e-6)
        assert np.isclose(norm(u), 1.0)

    def test_zero_vector_raises(self):
        with pytest.raises(AssertionError):
            orthogonal_vector([0, 0, 0])

    def test_1d_raises(self):
        with pytest.raises(AssertionError):
            orthogonal_vector([1])


class TestVectorProjection:
    def test_project_onto_x_axis(self):
        proj = vector_projection([3, 4, 0], [1, 0, 0])
        assert np.allclose(proj, [3, 0, 0])

    def test_parallel_vectors(self):
        proj = vector_projection([2, 0, 0], [5, 0, 0])
        assert np.allclose(proj, [2, 0, 0])

    def test_perpendicular_vectors(self):
        proj = vector_projection([0, 1, 0], [1, 0, 0])
        assert np.allclose(proj, [0, 0, 0])

    def test_zero_v2_raises(self):
        with pytest.raises(AssertionError):
            vector_projection([1, 0, 0], [0, 0, 0])


class TestProjectVectorOnPlane:
    def test_z_component_removed(self):
        result = project_vector_on_plane([1, 2, 3], [0, 0, 1])
        assert np.allclose(result, [1, 2, 0])

    def test_in_plane_vector_unchanged(self):
        result = project_vector_on_plane([1, 1, 0], [0, 0, 1])
        assert np.allclose(result, [1, 1, 0])

    def test_normal_projects_to_zero(self):
        result = project_vector_on_plane([0, 0, 5], [0, 0, 1])
        assert np.allclose(result, [0, 0, 0])


# ── Angle utilities ───────────────────────────────────────────────────────────

class TestAngleBetweenVectors:
    def test_parallel(self):
        a = angle_between_vectors([1, 0, 0], [2, 0, 0])
        assert np.isclose(a, 0.0)

    def test_perpendicular(self):
        a = angle_between_vectors([1, 0, 0], [0, 1, 0])
        assert np.isclose(a, np.pi / 2)

    def test_antiparallel(self):
        a = angle_between_vectors([1, 0, 0], [-1, 0, 0])
        assert np.isclose(a, np.pi)

    def test_degrees(self):
        a = angle_between_vectors([1, 0, 0], [0, 1, 0], degrees=True)
        assert np.isclose(a, 90.0)


# ── Projection utilities ──────────────────────────────────────────────────────

class TestProjectPointOnLine:
    def test_foot_of_perpendicular(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        proj = project_point_on_line([3, 5, 0], line)
        assert np.allclose(proj, [3, 0, 0])

    def test_point_on_line_unchanged(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        proj = project_point_on_line([7, 0, 0], line)
        assert np.allclose(proj, [7, 0, 0])


class TestProjectPointOnPlane:
    def test_above_xy_plane(self):
        plane = [0, 0, 1, 0]  # z=0
        proj = project_point_on_plane([1, 2, 5], plane)
        assert np.allclose(proj, [1, 2, 0])


# ── Distance point-to-line / point-to-plane ───────────────────────────────────

class TestDistancePointToLine:
    def test_known_distance(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        d = distance_point_to_line([0, 3, 4], line)
        assert np.isclose(d, 5.0)

    def test_point_on_line_zero(self):
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        d = distance_point_to_line([5, 0, 0], line)
        assert np.isclose(d, 0.0)


class TestDistancePointToPlane:
    def test_above_xy_plane(self):
        plane = [0, 0, 1, 0]
        d = distance_point_to_plane([0, 0, 5], plane)
        assert np.isclose(d, 5.0)

    def test_point_on_plane_zero(self):
        plane = [0, 0, 1, 0]
        d = distance_point_to_plane([1, 2, 0], plane)
        assert np.isclose(d, 0.0)

    def test_offset_plane(self):
        plane = [0, 0, 1, -3]  # z = 3
        d = distance_point_to_plane([0, 0, 0], plane)
        assert np.isclose(d, 3.0)


# ── Intersection utilities ────────────────────────────────────────────────────

class TestIntersectionBetweenLines:
    def test_3d_x_axis_y_axis(self):
        line1 = line_from_two_points([0, 0, 0], [1, 0, 0])
        line2 = line_from_two_points([0, 0, 0], [0, 1, 0])
        pt = intersection_between_lines(line1, line2)
        assert np.allclose(pt, [0, 0, 0])

    def test_3d_offset_intersection(self):
        line1 = line_from_two_points([0, 0, 1], [1, 0, 1])
        line2 = line_from_two_points([0, 0, 1], [0, 1, 1])
        pt = intersection_between_lines(line1, line2)
        assert np.allclose(pt, [0, 0, 1])

    def test_parallel_lines_raise(self):
        line1 = line_from_two_points([0, 0, 0], [1, 0, 0])
        line2 = line_from_two_points([0, 1, 0], [1, 1, 0])
        with pytest.raises(AssertionError):
            intersection_between_lines(line1, line2)


class TestLineFromPlanes:
    def test_xy_and_xz_planes(self):
        # xy-plane: z=0, xz-plane: y=0 → intersection is x-axis
        line = line_from_planes([0, 0, 1, 0], [0, 1, 0, 0])
        # direction should be along x
        dir_vec = np.abs(unit_vector(line[1]))
        assert np.allclose(dir_vec, [1, 0, 0])


class TestPointFromPlaneAndLine:
    def test_z_axis_hits_z3_plane(self):
        plane = [0, 0, 1, -3]   # z=3
        line = line_from_two_points([0, 0, 0], [0, 0, 1])
        pt = point_from_plane_and_line(plane, line)
        assert np.allclose(pt, [0, 0, 3])


# ── Circle ────────────────────────────────────────────────────────────────────

class TestCircleFromThreePoints:
    def test_xy_unit_circle(self):
        # circle_from_three_points returns (center, radius, plane)
        p1, p2, p3 = [1, 0, 0], [0, 1, 0], [-1, 0, 0]
        center, radius, plane = circle_from_three_points(p1, p2, p3)
        assert np.allclose(center, [0, 0, 0], atol=1e-6)
        assert np.isclose(radius, 1.0, atol=1e-6)


class TestPointCircleRelation:
    def test_inside(self):
        circle = circle_from_center_and_radius([0, 0, 0], 5.0)
        rel = point_circle_relation([1, 0, 0], circle)
        assert rel == PointCircleEnum.INSIDE

    def test_on(self):
        circle = circle_from_center_and_radius([0, 0, 0], 5.0)
        rel = point_circle_relation([5, 0, 0], circle)
        assert rel == PointCircleEnum.ON_BORDER

    def test_outside(self):
        circle = circle_from_center_and_radius([0, 0, 0], 5.0)
        rel = point_circle_relation([10, 0, 0], circle)
        assert rel == PointCircleEnum.OUTSIDE


# ── Sphere ────────────────────────────────────────────────────────────────────

class TestPointSphereRelation:
    # sphere is a tuple (center, radius)
    def test_inside(self):
        sphere = ([0, 0, 0], 5.0)
        assert point_sphere_relation([1, 0, 0], sphere) == PointShapeEnum.INSIDE

    def test_on(self):
        sphere = ([0, 0, 0], 5.0)
        assert point_sphere_relation([5, 0, 0], sphere) == PointShapeEnum.ON_BORDER

    def test_outside(self):
        sphere = ([0, 0, 0], 5.0)
        assert point_sphere_relation([10, 0, 0], sphere) == PointShapeEnum.OUTSIDE


# ── Coordinate conversions ───────────────────────────────────────────────────

class TestCartesianSphericalConversion:
    def test_roundtrip(self):
        points = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 1], [3, 4, 0],
        ]
        for pt in points:
            sph = cartesian_to_spherical(pt)
            cart = spherical_to_cartesian(sph)
            assert np.allclose(cart, pt, atol=1e-6), f"Roundtrip failed for {pt}"

    def test_known_point(self):
        # (1, 0, 0) → r=1, theta=0 (azimuth from x-axis), phi=pi/2 (zenith from z-axis)
        sph = cartesian_to_spherical([1, 0, 0])
        assert np.isclose(sph[0], 1.0)
        assert np.isclose(sph[1], 0.0, atol=1e-6)      # theta: azimuth in xy-plane
        assert np.isclose(sph[2], np.pi / 2, atol=1e-6)  # phi: angle from z-axis


# ── is_point_on_plane / is_line_on_plane ─────────────────────────────────────

class TestIsPointOnPlane:
    def test_on_plane(self):
        plane = [0, 0, 1, 0]
        assert is_point_on_plane([1, 2, 0], plane)

    def test_off_plane(self):
        plane = [0, 0, 1, 0]
        assert not is_point_on_plane([1, 2, 1], plane)


class TestIsLineOnPlane:
    def test_line_on_xy_plane(self):
        plane = [0, 0, 1, 0]
        line = line_from_two_points([0, 0, 0], [1, 0, 0])
        assert is_line_on_plane(line, plane)

    def test_line_not_on_plane(self):
        plane = [0, 0, 1, 0]
        line = line_from_two_points([0, 0, 0], [0, 0, 1])
        assert not is_line_on_plane(line, plane)


# ── Angle bisector ────────────────────────────────────────────────────────────

class TestAngleBisectorLineFromTwoLines:
    def test_2d_bisectors_perpendicular(self):
        # x-axis and y-axis: bisectors should be at 45° and 135°
        line1 = line_from_two_points([0, 0], [1, 0])
        line2 = line_from_two_points([0, 0], [0, 1])
        b1, b2 = angle_bisector_line_from_two_lines(line1, line2)
        # each bisector passes through origin
        assert np.isclose(b1[2], 0.0, atol=1e-6)
        assert np.isclose(b2[2], 0.0, atol=1e-6)

    def test_2d_parallel_raises(self):
        line1 = line_from_two_points([0, 0], [1, 0])
        line2 = line_from_two_points([0, 1], [1, 1])
        with pytest.raises(AssertionError):
            angle_bisector_line_from_two_lines(line1, line2)

    def test_3d_bisectors_through_intersection(self):
        line1 = line_from_two_points([0, 0, 0], [1, 0, 0])
        line2 = line_from_two_points([0, 0, 0], [0, 1, 0])
        b1, b2 = angle_bisector_line_from_two_lines(line1, line2)
        assert is_point_on_line([0, 0, 0], b1)
        assert is_point_on_line([0, 0, 0], b2)


class TestAngleBisectorPlaneFromTwoLines:
    def test_intersection_point_on_both_planes(self):
        line1 = line_from_two_points([0, 0, 0], [1, 0, 0])
        line2 = line_from_two_points([0, 0, 0], [0, 1, 0])
        p1, p2 = angle_bisector_plane_from_two_lines(line1, line2)
        for plane in [p1, p2]:
            val = np.dot(plane[:3], [0, 0, 0]) + plane[3]
            assert np.isclose(val, 0.0, atol=1e-6)


class TestAngleBisectorPlaneFromTwoPlanes:
    def test_returns_two_unit_planes(self):
        plane1 = [1, 0, 0, 0]   # yz-plane
        plane2 = [0, 1, 0, 0]   # xz-plane
        b1, b2 = angle_bisector_plane_from_two_planes(plane1, plane2)
        assert np.isclose(norm(b1[:3]), 1.0)
        assert np.isclose(norm(b2[:3]), 1.0)

    def test_parallel_planes_raise(self):
        with pytest.raises(AssertionError):
            angle_bisector_plane_from_two_planes([0, 0, 1, 0], [0, 0, 1, -1])


# ── vector_rotation ───────────────────────────────────────────────────────────

class TestVectorRotation:
    def test_90_degrees(self):
        result = vector_rotation([1, 0], np.pi / 2)
        assert np.allclose(result, [0, 1], atol=1e-6)

    def test_180_degrees(self):
        result = vector_rotation([1, 0], np.pi)
        assert np.allclose(result, [-1, 0], atol=1e-6)

    def test_zero_rotation(self):
        result = vector_rotation([3, 4], 0.0)
        assert np.allclose(result, [3, 4])

    def test_3d_raises(self):
        with pytest.raises(AssertionError):
            vector_rotation([1, 0, 0], np.pi / 2)


# ── project_line_on_plane ─────────────────────────────────────────────────────

class TestProjectLineOnPlane:
    def test_projected_point_on_plane(self):
        line = line_from_two_points([0, 0, 5], [1, 0, 5])
        plane = [0, 0, 1, 0]   # z=0
        proj = project_line_on_plane(line, plane)
        val = np.dot(plane[:3], proj[0]) + plane[3]
        assert np.isclose(val, 0.0, atol=1e-6)

    def test_direction_perpendicular_to_normal(self):
        line = line_from_two_points([0, 0, 5], [1, 0, 5])
        plane = [0, 0, 1, 0]
        proj = project_line_on_plane(line, plane)
        assert np.isclose(np.dot(proj[1], plane[:3]), 0.0, atol=1e-6)


# ── intersection_between_line_segments ───────────────────────────────────────

class TestIntersectionBetweenLineSegments:
    def test_crossing_segments(self):
        pt = intersection_between_line_segments([[0, 0], [2, 2]], [[0, 2], [2, 0]])
        assert np.allclose(pt, [1, 1])

    def test_non_crossing_segments_returns_nan(self):
        pt = intersection_between_line_segments([[0, 0], [1, 0]], [[2, 0], [3, 0]])
        assert np.all(np.isnan(pt))

    def test_parallel_segments_returns_nan(self):
        pt = intersection_between_line_segments([[0, 0], [1, 0]], [[0, 1], [1, 1]])
        assert np.all(np.isnan(pt))


# ── line_from_noisy_data ──────────────────────────────────────────────────────

class TestLineFromNoisyData:
    def test_2d_x_axis_recovery(self):
        rng = np.random.default_rng(0)
        pts = np.column_stack([np.linspace(0, 10, 20), rng.normal(0, 0.01, 20)])
        line = line_from_noisy_data(pts)
        assert np.isclose(norm(line[:2]), 1.0, atol=1e-6)

    def test_3d_direction_recovery(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 10, 30)
        pts = np.column_stack([t, rng.normal(0, 0.01, 30), rng.normal(0, 0.01, 30)])
        line = line_from_noisy_data(pts)
        # direction should be roughly along x-axis
        assert np.isclose(abs(unit_vector(line[1])[0]), 1.0, atol=0.05)


# ── point_from_three_planes ───────────────────────────────────────────────────

class TestPointFromThreePlanes:
    def test_axis_aligned(self):
        # x=1, y=2, z=3
        p1 = [1, 0, 0, -1]
        p2 = [0, 1, 0, -2]
        p3 = [0, 0, 1, -3]
        pt = point_from_three_planes(p1, p2, p3)
        assert np.allclose(pt, [1, 2, 3], atol=1e-6)


# ── circle_from_center_and_points ─────────────────────────────────────────────

class TestCircleFromCenterAndPoints:
    def test_2d(self):
        center, radius, _ = circle_from_center_and_points([0, 0], [3, 0])
        assert np.allclose(center, [0, 0])
        assert np.isclose(radius, 3.0)

    def test_3d(self):
        center, radius, _ = circle_from_center_and_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
        assert np.isclose(radius, 1.0)


# ── circle_from_center_and_radius ─────────────────────────────────────────────

class TestCircleFromCenterAndRadius:
    def test_2d(self):
        center, radius, _ = circle_from_center_and_radius([0, 0], 5.0)
        assert np.isclose(radius, 5.0)

    def test_3d(self):
        center, radius, plane = circle_from_center_and_radius([0, 0, 0], 3.0)
        assert np.isclose(radius, 3.0)
        assert len(plane) == 4

    def test_negative_radius_raises(self):
        with pytest.raises(AssertionError):
            circle_from_center_and_radius([0, 0], -1.0)


# ── circle_from_noisy_data ────────────────────────────────────────────────────

class TestCircleFromNoisyData:
    def test_unit_circle_recovery(self):
        rng = np.random.default_rng(42)
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        pts = np.column_stack([np.cos(angles) + rng.normal(0, 0.01, 30),
                               np.sin(angles) + rng.normal(0, 0.01, 30)])
        center, radius, _ = circle_from_noisy_data(pts)
        assert np.allclose(center, [0, 0], atol=0.05)
        assert np.isclose(radius, 1.0, atol=0.05)


# ── circle_coordinate_transform ───────────────────────────────────────────────

class TestCircleCoordinateTransform:
    def test_returns_4x4_matrix(self):
        plane = [0, 0, 1, 0]
        T = circle_coordinate_transform([0, 0, 0], plane)
        assert np.array(T).shape == (4, 4)

    def test_origin_maps_to_center(self):
        center = [1, 2, 3]
        plane = [0, 0, 1, -3]
        T = circle_coordinate_transform(center, plane)
        mapped = (T @ [0, 0, 0, 1])[:3]
        assert np.allclose(mapped, center, atol=1e-6)


# ── arc_from_center_and_endpoints ─────────────────────────────────────────────

class TestArcFromCenterAndEndpoints:
    def test_2d_radius(self):
        center, radius, thetas, T = arc_from_center_and_endpoints([0, 0], [1, 0], [0, 1])
        assert np.isclose(radius, 1.0)

    def test_3d_radius(self):
        center, radius, thetas, T = arc_from_center_and_endpoints(
            [0, 0, 0], [1, 0, 0], [0, 1, 0])
        assert np.isclose(radius, 1.0)
        assert T.shape == (4, 4)


# ── arc_from_three_points ─────────────────────────────────────────────────────

class TestArcFromThreePoints:
    def test_2d_unit_arc(self):
        center, radius, thetas, T = arc_from_three_points([1, 0], [0, 1], [-1, 0])
        assert np.isclose(radius, 1.0, atol=1e-6)

    def test_3d_unit_arc(self):
        # [1,0,0],[0,1,0],[0,0,1] lie on a sphere; radius = sqrt(2/3)
        center, radius, thetas, T = arc_from_three_points([1, 0, 0], [0, 1, 0], [0, 0, 1])
        assert np.isclose(radius, np.sqrt(2 / 3), atol=1e-6)
        assert T.shape == (4, 4)


# ── generate_points_on_circle ─────────────────────────────────────────────────

class TestGeneratePointsOnCircle:
    def test_2d_count_and_radius(self):
        center, radius, plane = circle_from_center_and_radius([0, 0], 3.0)
        pts = generate_points_on_circle(center, radius, plane, num=20)
        assert pts.shape == (20, 2)
        dists = [distance_between_points(p, center) for p in pts]
        assert np.allclose(dists, radius, atol=1e-6)

    def test_3d_count_and_radius(self):
        center, radius, plane = circle_from_center_and_radius([0, 0, 0], 2.0)
        pts = generate_points_on_circle(center, radius, plane, num=30)
        assert pts.shape == (30, 3)
        dists = [distance_between_points(p, center) for p in pts]
        assert np.allclose(dists, radius, atol=1e-6)


# ── generate_points_on_arc ────────────────────────────────────────────────────

class TestGeneratePointsOnArc:
    def test_2d_count_and_radius(self):
        arc = arc_from_three_points([1, 0], [0, 1], [-1, 0])
        center, radius, thetas, T = arc
        pts = generate_points_on_arc(center, radius, thetas, T, num=15)
        assert pts.shape == (15, 2)
        dists = [distance_between_points(p, center) for p in pts]
        assert np.allclose(dists, radius, atol=1e-6)


# ── intersection_between_line_and_circle ──────────────────────────────────────

class TestIntersectionBetweenLineAndCircle:
    def test_2d_two_intersections(self):
        circle = circle_from_center_and_radius([0, 0], 1.0)
        line = line_from_two_points([0, -2], [0, 2])    # y-axis: x=0
        pts = intersection_between_line_and_circle(line, circle)
        assert len(pts) == 2
        for p in pts:
            assert np.isclose(distance_between_points(p, [0, 0]), 1.0, atol=1e-6)

    def test_2d_tangent(self):
        circle = circle_from_center_and_radius([0, 0], 1.0)
        line = line_from_point_vector([1, 0], [0, 1])   # x=1 tangent
        pts = intersection_between_line_and_circle(line, circle)
        assert len(pts) == 1

    def test_2d_no_intersection(self):
        circle = circle_from_center_and_radius([0, 0], 1.0)
        line = line_from_point_vector([5, 0], [0, 1])   # x=5
        pts = intersection_between_line_and_circle(line, circle)
        assert len(pts) == 0


# ── is_point_on_arc ───────────────────────────────────────────────────────────

class TestIsPointOnArc:
    def test_endpoint_is_on_arc(self):
        arc = arc_from_three_points([1, 0], [0, 1], [-1, 0])
        assert is_point_on_arc([1, 0], arc)

    def test_off_arc_circle_not_on_arc(self):
        # quarter arc from (1,0) to (0,1), theta range [0, pi/2]
        arc = arc_from_three_points([1, 0], [np.cos(np.pi / 4), np.sin(np.pi / 4)], [0, 1])
        # (-1,0) is at angle pi from x-axis, outside [0, pi/2]
        assert not is_point_on_arc([-1, 0], arc)

    def test_interior_off_radius_not_on_arc(self):
        arc = arc_from_three_points([1, 0], [0, 1], [-1, 0])
        assert not is_point_on_arc([0, 0], arc)


# ── point_triangle_relation ───────────────────────────────────────────────────

class TestPointTriangleRelation:
    def test_inside(self):
        rel = point_triangle_relation([0.25, 0.25], [0, 0], [1, 0], [0, 1])
        assert rel == PointTriangleEnum.INSIDE

    def test_outside(self):
        rel = point_triangle_relation([1, 1], [0, 0], [1, 0], [0, 1])
        assert rel == PointTriangleEnum.OUTSIDE

    def test_on_border(self):
        rel = point_triangle_relation([0.5, 0], [0, 0], [1, 0], [0, 1])
        assert rel == PointTriangleEnum.ON_BORDER

    def test_3d_not_in_plane(self):
        rel = point_triangle_relation([0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0])
        assert rel == PointTriangleEnum.NOT_IN_PLANE


# ── point_cylinder_relation ───────────────────────────────────────────────────

class TestPointCylinderRelation:
    # cylinder: (bottom_center, radius, height, direction)
    def _cyl(self):
        return ([0, 0, 0], 1.0, 2.0, [0, 0, 1])

    def test_inside(self):
        rel = point_cylinder_relation([0, 0, 1], self._cyl())
        assert rel == PointShapeEnum.INSIDE

    def test_outside_radially(self):
        rel = point_cylinder_relation([5, 0, 1], self._cyl())
        assert rel == PointShapeEnum.OUTSIDE

    def test_outside_axially(self):
        rel = point_cylinder_relation([0, 0, 5], self._cyl())
        assert rel == PointShapeEnum.OUTSIDE


# ── spherical_cap_volume ──────────────────────────────────────────────────────

class TestSphericalCapVolume:
    def test_half_sphere(self):
        r = 2.0
        # half sphere: h = r
        v = spherical_cap_volume(r, r)
        expected = np.pi * r**3 * 2 / 3
        assert np.isclose(v, expected, rtol=1e-6)

    def test_zero_height(self):
        assert spherical_cap_volume(5.0, 0.0) == 0.0

    def test_invalid_height_raises(self):
        with pytest.raises(AssertionError):
            spherical_cap_volume(1.0, 2.0)   # h >= 2r


# ── overlap_volume_between_spheres ────────────────────────────────────────────

class TestOverlapVolumeBetweenSpheres:
    def test_no_overlap(self):
        s1 = ([0, 0, 0], 1.0)
        s2 = ([10, 0, 0], 1.0)
        assert overlap_volume_between_spheres(s1, s2) == 0

    def test_overlap_positive(self):
        s1 = ([0, 0, 0], 1.0)
        s2 = ([1, 0, 0], 1.0)
        assert overlap_volume_between_spheres(s1, s2) > 0

    def test_symmetry(self):
        s1 = ([0, 0, 0], 1.0)
        s2 = ([1, 0, 0], 1.0)
        assert np.isclose(overlap_volume_between_spheres(s1, s2),
                          overlap_volume_between_spheres(s2, s1))


# ── random_point_on_sphere ────────────────────────────────────────────────────

class TestRandomPointOnSphere:
    def test_single_point_on_surface(self):
        sphere = ([0, 0, 0], 3.0)
        p = random_point_on_sphere(sphere, num_points=1)
        assert np.isclose(distance_between_points(p, [0, 0, 0]), 3.0, atol=1e-6)

    def test_multiple_points_shape(self):
        sphere = ([1, 2, 3], 2.0)
        pts = random_point_on_sphere(sphere, num_points=10)
        assert pts.shape == (10, 3)
        for p in pts:
            assert np.isclose(distance_between_points(p, [1, 2, 3]), 2.0, atol=1e-6)


# ── distance_between_points_on_sphere ─────────────────────────────────────────

class TestDistanceBetweenPointsOnSphere:
    def test_quarter_great_circle(self):
        sphere = ([0, 0, 0], 1.0)
        # quarter great circle = pi/2 * r = pi/2
        d = distance_between_points_on_sphere([1, 0, 0], [0, 1, 0], sphere)
        assert np.isclose(d, np.pi / 2, atol=1e-6)

    def test_same_point_zero_distance(self):
        sphere = ([0, 0, 0], 2.0)
        d = distance_between_points_on_sphere([2, 0, 0], [2, 0, 0], sphere)
        assert np.isclose(d, 0.0, atol=1e-6)

    def test_antipodal_points(self):
        sphere = ([0, 0, 0], 1.0)
        d = distance_between_points_on_sphere([1, 0, 0], [-1, 0, 0], sphere)
        assert np.isclose(d, np.pi, atol=1e-6)


# ── parabolic_length / inv_parabolic_length ───────────────────────────────────

class TestParabolicLength:
    def test_symmetric_parabola(self):
        # y = x^2: length from -1 to 1 should equal length from 0 to 1 doubled
        coeff = [1, 0, 0]
        l_half = parabolic_length(coeff, 0, 1)
        l_full = parabolic_length(coeff, -1, 1)
        assert np.isclose(l_full, 2 * l_half, rtol=1e-6)

    def test_x1_greater_than_x2_swapped(self):
        coeff = [1, 0, 0]
        assert np.isclose(parabolic_length(coeff, 0, 2), parabolic_length(coeff, 2, 0))

    def test_zero_leading_coeff_raises(self):
        with pytest.raises(AssertionError):
            parabolic_length([0, 1, 0], 0, 1)

    def test_positive_length(self):
        assert parabolic_length([1, 0, 0], 0, 1) > 0


class TestInvParabolicLength:
    def test_roundtrip(self):
        coeff = [1, 0, 0]
        x1, target_len = 0.0, 1.0
        x2 = inv_parabolic_length(coeff, x1, target_len)
        assert np.isclose(parabolic_length(coeff, x1, x2), target_len, rtol=1e-6)

    def test_x2_greater_than_x1(self):
        coeff = [2, -1, 3]
        x2 = inv_parabolic_length(coeff, 0.0, 0.5)
        assert x2 > 0.0

    def test_nonpositive_length_raises(self):
        with pytest.raises(AssertionError):
            inv_parabolic_length([1, 0, 0], 0.0, 0.0)


# ── Triangle & Polygon Area ───────────────────────────────────────────────────

class TestTriangleArea:
    def test_2d(self):
        assert triangle_area([0, 0], [1, 0], [0, 1]) == 0.5

    def test_3d_xy(self):
        assert triangle_area([0, 0, 0], [1, 0, 0], [0, 1, 0]) == 0.5

    def test_3d(self):
        assert triangle_area([0, 0, 0], [3, 4, 0], [0, 0, 2]) == 5

    def test_high_dim(self):
        with pytest.raises(AssertionError):
            triangle_area([1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0])

    def test_mismatched_dims_raises(self):
        with pytest.raises(AssertionError):
            triangle_area([0, 0], [1, 2, 3], [4, 5, 6])

    def test_duplicated_points_2d(self):
        with pytest.raises(AssertionError):
            triangle_area([0, 0], [0, 0], [1, 2])

    def test_duplicated_points_3d(self):
        with pytest.raises(AssertionError):
            triangle_area([0, 0, 0], [0, 0, 0], [1, 2, 0])

    def test_colinear_2d(self):
        with pytest.raises(AssertionError):
            triangle_area([0, 0], [1, 2], [2, 4])

    def test_colinear_3d(self):
        with pytest.raises(AssertionError):
            triangle_area([0, 0, 0], [1, 2, 3], [2, 4, 6])


class TestPolygonArea:
    points = [
        [2, 0],
        [3, 1],
        [2, 2],
        [1, 2],
        [0, 1]
    ]

    def test_two_points(self):
        with pytest.raises(AssertionError):
            polygon_area([[1, 2], [3, 4]])

    def test_high_dim(self):
        with pytest.raises(AssertionError):
            polygon_area([[1, 2, 3, 4], [4, 3, 2, 1], [0, 0, 0, 0]])

    def test_downward_compatibility(self):
        points = [[0, 0], [1, 0], [0, 1]]
        assert polygon_area(points) == triangle_area(*points)

    def test_non_coplanar(self):
        with pytest.raises(AssertionError):
            polygon_area([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_2d(self):
        assert polygon_area(self.points) == 3.5

    def test_3d(self):
        points_3d = np.hstack([self.points, np.zeros((5, 1))])
        assert polygon_area(points_3d) == 3.5

    def test_unit_square(self):
        assert polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]]) == 1

    def test_composition(self):
        assert polygon_area(self.points) == sum([
            triangle_area(self.points[0], self.points[1], self.points[2]),
            triangle_area(self.points[0], self.points[2], self.points[3]),
            triangle_area(self.points[0], self.points[3], self.points[4]),
        ])
