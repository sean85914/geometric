import pytest
import numpy as np
from geometric.ellipse import Ellipse


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


# ── sample_points ─────────────────────────────────────────────────────────────

class TestSamplePoints:
    def test_shape(self):
        pts = Ellipse(1 / 4, 0, 1, 0, 0, -1).sample_points(100)
        assert pts.shape == (100, 2)

    def test_points_on_ellipse(self):
        e = Ellipse(1 / 4, 0, 1, 0, 0, -1)
        pts = e.sample_points(50)
        for p in pts:
            val = e.A * p[0]**2 + e.B * p[0] * p[1] + e.C * p[1]**2 + e.D * p[0] + e.E * p[1] + e.F
            assert np.isclose(val, 0.0, atol=1e-6)

    def test_rotated_ellipse_points_on_curve(self):
        e = Ellipse(5 / 8, -3 / 4, 5 / 8, 0, 0, -1)
        pts = e.sample_points(50)
        for p in pts:
            val = e.A * p[0]**2 + e.B * p[0] * p[1] + e.C * p[1]**2 + e.D * p[0] + e.E * p[1] + e.F
            assert np.isclose(val, 0.0, atol=1e-6)


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
