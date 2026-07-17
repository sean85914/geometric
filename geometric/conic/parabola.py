from dataclasses import dataclass
import numpy as np
from geometric.conic import Conic


@dataclass
class Parabola(Conic):
    def __post_init__(self):
        if self.A < 0:
            self.A, self.B, self.C, self.D, self.E, self.F = np.array(
                [self.A, self.B, self.C, self.D, self.E, self.F]
            ) * -1  # make sure leading coefficient is positive

        D, E, F = self.D, self.E, self.F
        assert np.isclose(self.discriminant, 0, atol=1e-5), f"Discriminent must be 0, got {self.discriminant}"
        self._form_M()
        # b * Y = a * X^2
        values, vectors = np.linalg.eigh(self.M[:2, :2])
        indices = np.argsort(np.abs(values))[::-1]
        self.a = values[indices[0]]  # lambda
        self.T = np.eye(3)
        self.T[:2, :2] = vectors[:, indices]
        if np.linalg.det(self.T[:2, :2]) < 0:
            self.T[:2, 1] *= -1  # fix handedness
        # Constraint theta \in (-pi/2, pi/2]
        if self.T[0, 0] < 0 or (np.isclose(self.T[0, 0], 0, atol=1e-5) and self.T[1, 0] < 0):
            self.T[:2, :2] *= -1
        K = np.array([D, E])
        D_prime, E_prime = K @ self.T[:2, :2]
        h_prime = -D_prime / (2 * self.a)
        k_prime = (-F + D_prime**2 / (4 * self.a)) / E_prime
        self.b = -E_prime
        self.T[:2, 2] = self.T[:2, :2] @ [h_prime, k_prime]
        self.T_inv = np.linalg.inv(self.T)
        self.theta = self._compute_theta()
        self.eccentricity = 1.0  # Constant

    @property
    def vertex(self):
        return self.T[:2, 2]

    @property
    def focus(self):
        c = self.b / (4 * self.a)
        return (self.T @ [0, c, 1])[:2]

    @property
    def directrix(self):
        c = self.b / (4 * self.a)
        h, k = self.T[:2, 2]
        sin = np.sin(self.theta)
        cos = np.cos(self.theta)
        return np.array([-sin, cos, sin * h - cos * k + c])

    def parametric(self, t):
        scalar_input = np.ndim(t) == 0
        t = np.atleast_1d(t).astype(float)
        x, y = t, self.a * t**2 / self.b
        p = np.vstack([x, y, [1] * len(t)])
        res = (self.T @ p).T[:, :2]
        if scalar_input:
            return res[0]
        return res

    def closest_point(self, point):
        assert np.atleast_2d(point).shape[0] == 1
        point = np.asarray(point, dtype=float)
        m = self.a / self.b
        all_roots = np.roots([
            2 - m**2,
            0,
            1 - 2 * m * point[1],
            -point[0]
        ])
        real_roots_mask = np.abs(all_roots.imag) < 1e-5
        real_roots = all_roots[real_roots_mask].real

        ps = self.parametric(real_roots)  # global
        dists = np.linalg.norm(ps - point, axis=1)
        return ps[np.argmin(dists)]

    def arc_length(self, t1, t2):
        ...
