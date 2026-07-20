from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Conic(ABC):
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float

    M: np.ndarray = field(init=False, repr=False)
    T: np.ndarray = field(init=False, repr=False)  # local -> global
    T_inv: np.ndarray = field(init=False, repr=False)  # global -> local

    a: float = field(init=False)
    b: float = field(init=False)
    theta: float = field(init=False)
    eccentricity: float = field(init=False)

    def _form_M(self):
        self.M = np.array([
            [self.A, self.B / 2, self.D / 2],
            [self.B / 2, self.C, self.E / 2],
            [self.D / 2, self.E / 2, self.F]
        ])

    @property
    def discriminant(self):
        return self.B**2 - 4 * self.A * self.C

    @property
    @abstractmethod
    def foci(self):
        pass

    @property
    @abstractmethod
    def directrix(self):
        pass

    def _solve_center(self):
        assert not np.isclose(np.linalg.det(self.M[:2, :2]), 0, atol=1e-5), 'Matrix is singular'
        h, k = np.linalg.solve(2 * self.M[:2, :2], [-self.D, -self.E])
        return h, k

    def evaluate(self, points):
        """Evaluate the general conic equation at one or more points.

        Arguments:
            points (array-like): A single 2D point ``[x, y]``, or an ``(N, 2)``
                array of points, in the global frame.

        Returns:
            float or numpy.ndarray: The conic equation value(s). Negative
                inside the ellipse, positive outside, zero on the boundary.

        Raises:
            AssertionError: If ``points`` is not a single 2D point or an
                ``(N, 2)`` array.
        """
        points = np.atleast_2d(points)  # (N, 2)
        assert points.shape[1] == 2
        points_extend = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 3)
        results = np.sum((points_extend @ self.M) * points_extend, axis=1)
        if points.shape[0] == 1:
            return results.item()
        return results

    def _compute_theta(self):
        return np.arctan2(self.T[1, 0], self.T[0, 0])

    @abstractmethod
    def parametric(self, t):
        pass

    @abstractmethod
    def closest_point(self, point):
        pass

    @classmethod
    def from_points(cls, points):
        ...
