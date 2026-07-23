import pytest
import numpy as np
from geometric.utils import solve_trig_linear_equation


class TestUtils:
    def test_zeros(self):
        with pytest.raises(ValueError):
            solve_trig_linear_equation(0, 0, 3)

    def test_no_sol(self):
        with pytest.raises(ValueError):
            solve_trig_linear_equation(1, 2, 3)

    def test_is_sols(self):
        A, B, C = 3, 2, 1
        thetas = solve_trig_linear_equation(A, B, C)
        results = []
        for theta in thetas:
            results.append(A * np.sin(theta) + B * np.cos(theta))
        assert np.allclose(results, C, atol=1e-5)

    def test_one_sol(self):
        A, B, C = 3, 4, 5
        thetas = solve_trig_linear_equation(A, B, C)
        assert np.isclose(thetas[0], thetas[1], atol=1e-5)
