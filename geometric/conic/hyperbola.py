from dataclasses import dataclass
import numpy as np
from geometric import line_from_point_vector
from geometric.conic import Conic


@dataclass
class Hyperbola(Conic):
    def __post_init__(self):
        if self.A < 0:
            self.A, self.B, self.C, self.D, self.E, self.F = np.array(
                [self.A, self.B, self.C, self.D, self.E, self.F]
            ) * -1  # make sure leading coefficient is positive

        assert self.discriminant > 0, f"Discriminent must greater than 0, got {self.discriminant}"
        self._form_M()
        h, k = self._solve_center()

        F_prime = self.F + np.array([self.D, self.E]) @ np.array([h, k]) / 2
        values, vectors = np.linalg.eigh(self.M[:2, :2])

        if F_prime * values[0] < 0:
            lambda_1, lambda_2 = values
            indices = [0, 1]
        else:
            lambda_2, lambda_1 = values
            indices = [1, 0]
        self.T = np.eye(3)
        self.T[:2, :2] = vectors[:, indices]
        self.T[:2, 2] = h, k
        self.T_inv = np.linalg.inv(self.T)
        self.theta = self._compute_theta()

        self.a = np.sqrt(-F_prime / lambda_1)
        self.b = np.sqrt(F_prime / lambda_2)
        self.eccentricity = np.sqrt(1 + self.b**2 / self.a**2)

    @property
    def center(self):
        return self.T[:2, 2]

    @property
    def foci(self):
        c = self.a * self.eccentricity
        local = np.array([[c, 0, 1], [-c, 0, 1]]).T
        return (self.T @ local)[:2].T

    @property
    def directrix(self):
        vec_global = (self.T @ [0.0, 1.0, 0.0])[:2]
        p_local = np.array([
            [self.a / self.eccentricity, 0.0, 1.0],
            [-self.a / self.eccentricity, 0.0, 1.0]
        ])
        p_global = (self.T @ p_local.T).T[:, :2]
        return np.array([
            line_from_point_vector(p_global[0], vec_global),
            line_from_point_vector(p_global[1], vec_global)
        ])

    def parametric(self, theta):
        scalar_input = np.ndim(theta) == 0
        theta = np.atleast_1d(theta).astype(float)
        local = np.vstack([self.a / np.cos(theta), self.b * np.tan(theta), np.ones_like(theta)])
        points = (self.T @ local)[:2].T
        return points[0] if scalar_input else points

    def closest_point(self, point):
        assert np.atleast_2d(point).shape[0] == 1
        point = np.asarray(point, dtype=float)
        if np.isclose(self.evaluate(point), 0, atol=1e-5):
            return point
        point_local = (self.T_inv @ [point[0], point[1], 1.0])[:2]
        c2 = self.a**2 + self.b**2
        # Solve tan\theta
        all_roots = np.roots([
            c2**2,
            -2 * self.b * point_local[1] * c2,
            c2**2 + self.b**2 * point_local[1]**2 - self.a**2 * point_local[0]**2,
            -2 * self.b * point_local[1] * c2,
            self.b**2 * point_local[1]**2
        ])
        real_roots_mask = np.abs(all_roots.imag) < 1e-5
        real_roots = all_roots[real_roots_mask].real
        theta = np.arctan(real_roots)
        thetas = np.concatenate([theta, theta + np.pi])
        ps = self.parametric(thetas)  # global
        dists = np.linalg.norm(ps - point, axis=1)
        return ps[np.argmin(dists)]

    @property
    def asymptote(self):
        vec_1 = (self.T @ [self.a, self.b, 0.0])[:2]
        vec_2 = (self.T @ [-self.a, self.b, 0.0])[:2]
        line_1 = line_from_point_vector(self.T[:2, 2], vec_1)
        line_2 = line_from_point_vector(self.T[:2, 2], vec_2)
        return [line_1, line_2]
