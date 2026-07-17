from dataclasses import dataclass
import numpy as np
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

    def parametric(self, theta):
        scalar_input = np.ndim(theta) == 0
        theta = np.atleast_1d(theta).astype(float)
        local = np.vstack([self.a / np.cos(theta), self.b * np.tan(theta), np.ones_like(theta)])
        points = (self.T @ local)[:2].T
        return points[0] if scalar_input else points

    def closest_point(self, point):
        raise NotImplementedError()
