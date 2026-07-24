import pytest
import numpy as np
from geometric.utils import solve_trig_linear_equation


def test_zeros():
    with pytest.raises(ValueError):
        solve_trig_linear_equation(0, 0, 3)


def test_no_sol():
    with pytest.raises(ValueError):
        solve_trig_linear_equation(1, 2, 3)


@pytest.mark.parametrize("A, B, C", [
    (3, 2, 1),
    (3, 2, -1),  # negative C
    (-3, 2, 1),  # negative A
    (3, -2, 1),  # negative B
    (-3, -2, -1),  # all negative
    (0, 2, 1),  # pure cos
    (3, 0, 1),  # pure sin
    (3, 2, 0),  # zero C
    (2, 3, 3.6),  # |C| near K
    (3, 4, 5),  # C / K = 1
])
def test_solutions_satisfy(A, B, C):
    thetas = solve_trig_linear_equation(A, B, C)
    for t in thetas:
        assert np.isclose(A * np.sin(t) + B * np.cos(t), C, atol=1e-6)


def test_near_tangent_within_tolerance_no_raise():
    # C/K = 1.000005，in tolerance
    A, B = 3, 4
    C = 5 * (1 + 5e-6)
    thetas = solve_trig_linear_equation(A, B, C)
    assert np.allclose(thetas[0], thetas[1], atol=1e-2)


def test_just_over_tolerance_raises():
    # C/K = 1.00005，out of tolerance
    with pytest.raises(ValueError):
        solve_trig_linear_equation(3, 4, 5 * (1 + 5e-5))
