from dataclasses import dataclass, field
import numpy as np


@dataclass
class Parabola:
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float

    M: np.ndarray = field(init=False)
    a: float = field(init=False)
    lambda_: float = field(init=False)
    # Y = a / lambda * X^2
    T: np.ndarray = field(init=False)  # local -> global
    T_inv: np.ndarray = field(init=False)  # global -> local
    theta: float = field(init=False)

    def __post_init__(self):
        if self.A < 0:
            self.A, self.B, self.C, self.D, self.E, self.F = np.array(
                [self.A, self.B, self.C, self.D, self.E, self.F]
            ) * -1  # make sure leading coefficient is positive

        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        discriminant = B**2 - 4 * A * C
        assert np.isclose(discriminant, 0, atol=1e-5), f"Discriminent must be 0, got {discriminant}"

        self.M = np.array([
            [A, B / 2],
            [B / 2, C]
        ])

        self.theta = 1 / 2 * np.arctan2(B, A - C)
        self.a = A + C
        c = np.cos(self.theta)
        s = np.sin(self.theta)

        self.lambda_ = D * s - E * c

        alpha = (D * c + E * s) / (-2 * self.a)
        beta = ((D * c + E * s)**2 / (4 * self.a) - F) / self.lambda_

        h = alpha * c + beta * s
        k = alpha * s - beta * c

        self.T = np.array([
            [c, -s, h],
            [s, c, k],
            [0.0, 0.0, 1.0]
        ])
        self.T_inv = np.linalg.inv(self.T)

    @property
    def vertex(self):
        return self.T[:2, 2]

    @property
    def focus(self):
        c = self.lambda_ / (4 * self.a)
        return (self.T @ [0, c, 1])[:2]

    @property
    def directrix(self):
        c = self.lambda_ / (4 * self.a)
        h, k = self.T[:2, 2]
        sin = np.sin(self.theta)
        cos = np.cos(self.theta)
        return np.array([-sin, cos, sin * h - cos * k + c])

    def evaluate(self, points):
        points = np.atleast_2d(points)  # (N, 2)
        assert points.shape[1] == 2
        results = np.sum((points @ self.M) * points, axis=1) + points @ np.array([self.D, self.E]) + self.F
        if points.shape[0] == 1:
            return results.item()
        return results

    def parametric(self, t):
        scalar_input = np.ndim(t) == 0
        t = np.atleast_1d(t).astype(float)
        x, y = t, self.a * t**2 / self.lambda_
        p = np.vstack([x, y, [1] * len(t)])
        res = (self.T @ p).T[:, :2]
        if scalar_input:
            return res[0]
        return res

    def closest_point(self, point):
        assert np.atleast_2d(point).shape[0] == 1
        point = np.asarray(point, dtype=float)
        m = self.a / self.lambda_
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

    @staticmethod
    def from_points(points):
        ...
