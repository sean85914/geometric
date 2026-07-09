import warnings
import pytest
import numpy as np
from geometric import Ellipse


# ── Construction ──────────────────────────────────────────────────────────────

class TestEllipseConstruction:
    def test_axis_aligned_x(self):
        # x^2/4 + y^2 = 1: long axis along x
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.isclose(e.a, 2.0)
        assert np.isclose(e.b, 1.0)
        assert np.allclose(e.center, [0, 0])
        assert np.isclose(e.theta, 0.0)

    def test_axis_aligned_y(self):
        # x^2 + y^2/4 = 1: long axis along y
        e = Ellipse(1, 0, 1 / 4, 0, 0, -1)
        assert np.isclose(e.a, 2.0)
        assert np.isclose(e.b, 1.0)
        assert np.isclose(abs(e.theta), np.pi / 2)

    def test_translated(self):
        # (x-1)^2/4 + (y-2)^2= 1 → A=1/4, B=0, C=1, D=-1/2, E=-4, F=13/4
        e = Ellipse(1 / 4, 0, 1, -1 / 2, -4, 13 / 4)
        assert np.allclose(e.center, [1, 2])
        assert np.isclose(e.a, 2.0)
        assert np.isclose(e.b, 1.0)

    def test_rotated_45deg(self):
        # a=2, b=1, theta=pi/4, center=(0,0)
        # A = C = 5/8, B = -3/4, F=-1
        e = Ellipse(5 / 8, -3 / 4, 5 / 8, 0, 0, -1)
        assert np.isclose(e.a, 2.0, atol=1e-6)
        assert np.isclose(e.b, 1.0, atol=1e-6)
        assert np.isclose(e.theta, np.pi / 4, atol=1e-6)

    def test_negative_leading_coeff(self):
        # Negating all coefficients should give the same ellipse
        e1 = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        e2 = Ellipse(-1 / 4, 0, -1, 0, 0, 1)
        assert np.isclose(e1.a, e2.a)
        assert np.isclose(e1.b, e2.b)
        assert np.allclose(e1.center, e2.center)

    def test_circle_is_valid(self):
        # Circle: a == b, eccentricity == 0
        e = Ellipse(1, 0, 1, 0, 0, -1)
        assert np.isclose(e.a, 1.0)
        assert np.isclose(e.b, 1.0)
        assert np.isclose(e.eccentricity, 0.0)

    def test_hyperbola_raises(self):
        with pytest.raises(AssertionError):
            Ellipse(1, 0, -1, 0, 0, -1)  # x^2 - y^2 = 1

    def test_parabola_raises(self):
        with pytest.raises(AssertionError):
            Ellipse(1, 2, 1, 0, 0, -1)   # B^2 - 4AC = 0


# ── center property ───────────────────────────────────────────────────────────

class TestCenter:
    def test_origin(self):
        assert np.allclose(Ellipse(1 / 4, 0, 1, 0, 0, -1).center, [0, 0])

    def test_translated(self):
        assert np.allclose(Ellipse(1 / 4, 0, 1, -1 / 2, -4, 13 / 4).center, [1, 2])


# ── Area ──────────────────────────────────────────────────────────────────────

class TestArea:
    def test_area(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)  # a=2, b=1
        assert np.isclose(e.area, 2 * np.pi)

    def test_circle_area(self):
        e = Ellipse(1, 0, 1, 0, 0, -1)   # r=1
        assert np.isclose(e.area, np.pi)

    def test_area_positive(self):
        assert Ellipse(1 / 4, 0, 1, 0, 0, -1).area > 0


# ── Perimeter ─────────────────────────────────────────────────────────────────

class TestPerimeter:
    def test_circle_perimeter(self):
        # E(0) = pi/2, so 4 * r * pi/2 = 2 * pi * r
        e = Ellipse(1, 0, 1, 0, 0, -1)
        assert np.isclose(e.perimeter, 2 * np.pi, atol=1e-6)

    def test_perimeter_positive(self):
        assert Ellipse(1 / 4, 0, 1, 0, 0, -1).perimeter > 0

    def test_perimeter_greater_than_major_axis(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert e.perimeter > 2 * e.a


# ── Arc length ────────────────────────────────────────────────────────────────

class TestArcLength:
    def test_full_arc_equals_perimeter(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.isclose(e.arc_length(0, 2 * np.pi), e.perimeter, rtol=1e-5)

    def test_arc_length_positive(self):
        assert Ellipse(1 / 4, 0, 1, 0, 0, -1).arc_length(0, np.pi / 2) > 0

    def test_arc_length_order_independent(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.isclose(e.arc_length(0, np.pi / 2), e.arc_length(np.pi / 2, 0))

    def test_half_arc(self):
        # Full arc / 2 == arc from 0 to pi (by symmetry of ellipse)
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.isclose(e.arc_length(0, np.pi), e.perimeter / 2, rtol=1e-6)


# ── Sector area ───────────────────────────────────────────────────────────────

class TestSectorArea:
    def test_full_sector_equals_area(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.isclose(e.sector_area(0, 2 * np.pi), e.area)

    def test_half_sector(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.isclose(e.sector_area(0, np.pi), e.area / 2)

    def test_sector_area_positive(self):
        assert Ellipse(1 / 4, 0, 1, 0, 0, -1).sector_area(0, np.pi / 4) > 0

    def test_sector_area_order_independent(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.isclose(e.sector_area(0, np.pi / 2), e.sector_area(np.pi / 2, 0))


# ── evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_on_curve_is_zero(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)  # x^2/4 + y^2 = 1
        assert np.isclose(e.evaluate([2, 0]), 0)
        assert np.isclose(e.evaluate([0, 1]), 0)

    def test_center_is_negative(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert e.evaluate([0, 0]) < 0

    def test_far_point_is_positive(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert e.evaluate([100, 100]) > 0

    def test_single_point_returns_scalar(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        result = e.evaluate([2, 0])
        assert isinstance(result, float)

    def test_batch_returns_correct_shape(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        pts = np.array([[2, 0], [0, 1], [0, 0]])
        result = e.evaluate(pts)
        assert result.shape == (3,)

    def test_rotated_ellipse_on_curve(self):
        e = Ellipse(5 / 8, -3 / 4, 5 / 8, 0, 0, -1)
        pts = e.sample_points(20)
        assert np.allclose(e.evaluate(pts), 0)


# ── sample_points ─────────────────────────────────────────────────────────────

class TestSamplePoints:
    def test_shape(self):
        pts = Ellipse(1 / 4, 0, 1, 0, 0, -1).sample_points(100)
        assert pts.shape == (100, 2)

    def test_points_on_ellipse(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        pts = e.sample_points(50)
        assert np.allclose(e.evaluate(pts), 0)

    def test_rotated_ellipse_points_on_curve(self):
        e = Ellipse(5 / 8, -3 / 4, 5 / 8, 0, 0, -1)
        pts = e.sample_points(50)
        assert np.allclose(e.evaluate(pts), 0)


# ── from_points ───────────────────────────────────────────────────────────────

class TestFromPoints:
    def test_roundtrip_axis_aligned(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        pts = e.sample_points(50)
        e2 = Ellipse.from_points(pts)
        assert np.isclose(e2.a, e.a, rtol=1e-3)
        assert np.isclose(e2.b, e.b, rtol=1e-3)
        assert np.allclose(e2.center, e.center, atol=1e-3)

    def test_roundtrip_rotated(self):
        e = Ellipse(5 / 8, -3 / 4, 5 / 8, 0, 0, -1)
        pts = e.sample_points(50)
        e2 = Ellipse.from_points(pts)
        assert np.isclose(e2.a, e.a, rtol=1e-3)
        assert np.isclose(e2.b, e.b, rtol=1e-3)

    def test_too_few_points_raises(self):
        with pytest.raises(AssertionError):
            Ellipse.from_points([[1, 0]] * 5)

    def test_wrong_dim_raises(self):
        with pytest.raises(AssertionError):
            Ellipse.from_points([[1, 0, 0]] * 10)


# ── parametric ────────────────────────────────────────────────────────────────

class TestParametric:
    def test_scalar_theta_returns_point(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)  # a=2, b=1
        assert np.allclose(e.parametric(0), [2, 0])
        assert np.allclose(e.parametric(np.pi / 2), [0, 1])

    def test_batch_theta_returns_array(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        pts = e.parametric([0, np.pi / 2])
        assert pts.shape == (2, 2)
        assert np.allclose(pts, [[2, 0], [0, 1]])

    def test_translated_center(self):
        e = Ellipse(1 / 4, 0, 1, -1 / 2, -4, 13 / 4)  # center (1, 2)
        assert np.allclose(e.parametric(0), [1 + 2, 2])

    def test_points_lie_on_curve(self):
        e = Ellipse(5 / 8, -3 / 4, 5 / 8, 0, 0, -1)  # rotated ellipse
        thetas = np.linspace(0, 2 * np.pi, 30)
        pts = e.parametric(thetas)
        assert np.allclose(e.evaluate(pts), 0, atol=1e-6)


# ── closest_point ─────────────────────────────────────────────────────────────

class TestClosestPoint:
    def test_circle_point_outside_on_axis(self):
        e = Ellipse(1, 0, 1, 0, 0, -1)  # unit circle at origin
        assert np.allclose(e.closest_point([5, 0]), [1, 0])

    def test_circle_point_outside_general_direction(self):
        e = Ellipse(1, 0, 1, 0, 0, -1)
        assert np.allclose(e.closest_point([3, 4]), [0.6, 0.8])

    def test_circle_point_inside(self):
        e = Ellipse(1, 0, 1, 0, 0, -1)
        assert np.allclose(e.closest_point([0.5, 0]), [1, 0])

    def test_circle_point_on_local_axis_no_warning(self):
        # Regression: a == b makes c2 == 0. A point on the (arbitrary) local
        # x-axis that ISN'T the center used to divide by zero (c2) inside
        # the Y1 == 0 branch. Promote warnings to errors to catch it.
        e = Ellipse(1, 0, 1, 0, 0, -1)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cp = e.closest_point([5, 0])
        assert np.allclose(cp, [1, 0])

    def test_circle_center_is_degenerate_and_warns(self):
        e = Ellipse(1, 0, 1, 0, 0, -1)
        with pytest.warns(UserWarning):
            cp = e.closest_point([0, 0])
        assert np.isclose(e.evaluate(cp), 0)       # returned point lies on the circle
        assert np.isclose(np.linalg.norm(cp), 1)   # distance from center == radius

    def test_point_on_curve_returns_itself(self):
        e = Ellipse(5 / 8, -3 / 4, 5 / 8, 0, 0, -1)  # rotated ellipse
        pts = e.sample_points(20)
        for p in pts:
            assert np.allclose(e.closest_point(p), p, atol=1e-4)

    def test_ellipse_point_on_major_axis(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)  # a=2, b=1
        assert np.allclose(e.closest_point([10, 0]), [2, 0])

    def test_ellipse_point_on_minor_axis(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.allclose(e.closest_point([0, 10]), [0, 1])

    def test_missing_theta_pi_root_regression(self):
        # Regression: point far on the -X side, Y1=0. Before the theta=pi
        # patch, the tan(theta/2) substitution silently dropped this root
        # and returned the far vertex (2, 0) instead of the near one.
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        assert np.allclose(e.closest_point([-10, 0]), [-2, 0])

    def test_matches_brute_force_search(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        p = np.array([3, 1.5])
        thetas = np.linspace(0, 2 * np.pi, 200000)
        candidates = e.parametric(thetas)
        brute_dist = np.linalg.norm(candidates - p, axis=1).min()
        actual_dist = np.linalg.norm(e.closest_point(p) - p)
        assert np.isclose(actual_dist, brute_dist, atol=1e-3)
