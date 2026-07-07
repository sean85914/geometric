from dataclasses import dataclass, field
import warnings
import numpy as np
from scipy.special import ellipe, ellipeinc
from .geometric import unit_vector


@dataclass
class Ellipse:
    """Represents a 2D ellipse defined by its general conic form.

    The ellipse is defined by the equation:

    .. math::

        Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0

    where :math:`B^2 - 4AC < 0`. If ``A < 0``, all coefficients are negated
    automatically so that the leading coefficient is always positive.

    On construction, the standard parameters are derived:

    - ``a``: semi-major axis length
    - ``b``: semi-minor axis length
    - ``theta``: rotation angle of the major axis from the x-axis (radians)
    - ``T``: 3×3 homogeneous transform from the ellipse's local frame to the
      global frame (columns are the major-axis direction, minor-axis direction,
      and the center)
    - ``eccentricity``: :math:`e = \\sqrt{1 - b^2/a^2}`

    Example:
        >>> e = Ellipse(1/4, 0, 1, 0, 0, -1)  # x^2/4 + y^2 = 1
        >>> e.a, e.b
        (2.0, 1.0)
    """

    A: float
    B: float
    C: float
    D: float
    E: float
    F: float

    M: np.ndarray = field(init=False)
    a: float = field(init=False)
    b: float = field(init=False)
    T: np.ndarray = field(init=False)  # local -> global
    T_inv: np.ndarray = field(init=False)  # global -> local
    theta: float = field(init=False)
    eccentricity: float = field(init=False)

    def __post_init__(self):
        if self.A < 0:
            self.A, self.B, self.C, self.D, self.E, self.F = np.array(
                [self.A, self.B, self.C, self.D, self.E, self.F]
            ) * -1  # make sure leading coefficient is positive
        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        discriminant = B**2 - 4 * A * C
        assert discriminant < 0, f"Discriminent must less than 0, got {discriminant}"

        self.M = np.array([
            [A, B / 2],
            [B / 2, C]
        ])
        h, k = np.linalg.solve(2 * self.M, [-D, -E])

        K = np.sqrt((A - C)**2 + B**2)
        v1 = (A + C - K) / 2
        v2 = (A + C + K) / 2
        if np.isclose(B, 0, atol=1e-5):
            ev1 = np.array([1.0, 0.0])
            ev2 = np.array([0.0, 1.0])
            if A > C:
                ev1, ev2 = ev2, -ev1
        else:
            ev1 = unit_vector(np.array([1, B / (A - C - K)]))
            ev2 = unit_vector(np.array([1, B / (A - C + K)]))

        self.T = np.eye(3)
        self.T[:2, :2] = np.vstack([ev1, ev2]).T
        self.T[:2, 2] = h, k
        self.T_inv = np.linalg.inv(self.T)
        self.theta = np.arctan2(self.T[1, 0], self.T[0, 0])

        F_prime = np.dot([A, B, C, D, E, F], [h**2, h * k, k**2, h, k, 1])
        assert -F_prime / v1 > 0, 'Not a real ellipse'

        self.a = np.sqrt(-F_prime / v1)
        self.b = np.sqrt(-F_prime / v2)
        self.eccentricity = np.sqrt(1 - self.b**2 / self.a**2)
        # x^2 / a^2 + y^2 / b^2 = 1

    @property
    def center(self):
        """The center of the ellipse.

        Returns:
            numpy.ndarray: A ``(2,)`` array ``[h, k]``.
        """
        return self.T[:2, 2]

    def sample_points(self, n):
        """Sample random points uniformly distributed in parameter angle on the ellipse.

        Arguments:
            n (int): Number of points to sample.

        Returns:
            numpy.ndarray: An ``(n, 2)`` array of 2D points on the ellipse boundary.
        """
        thetas = np.random.uniform(0, 2 * np.pi, n)
        return self.parametric(thetas)

    @property
    def area(self):
        """Area of the ellipse.

        Returns:
            float: :math:`\\pi ab`.
        """
        return np.pi * self.a * self.b

    @property
    def perimeter(self):
        """Perimeter of the ellipse via the complete elliptic integral of the second kind.

        Returns:
            float: :math:`4a \\cdot E(e^2)`, where :math:`E` is the complete elliptic
            integral of the second kind and :math:`e` is the eccentricity.
        """
        return 4 * self.a * ellipe(self.eccentricity**2)

    def arc_length(self, theta1, theta2):
        """Arc length along the ellipse between two parameter angles.

        Uses the incomplete elliptic integral of the second kind:
        :math:`a \\cdot [E(\\theta_2, e^2) - E(\\theta_1, e^2)]`.

        Arguments:
            theta1 (float): Start parameter angle in radians.
            theta2 (float): End parameter angle in radians.

        Returns:
            float: Arc length from ``theta1`` to ``theta2``.
        """
        e2 = self.eccentricity**2
        return self.a * np.abs(ellipeinc(theta2, e2) - ellipeinc(theta1, e2))

    def sector_area(self, theta1, theta2):
        """Area of the ellipse sector swept from ``theta1`` to ``theta2``.

        The sector is the region bounded by the two radii from the center to
        the ellipse at ``theta1`` and ``theta2``, and the arc between them.
        Derived via Green's theorem:

        .. math::

            A = \\frac{ab}{2} |\\theta_2 - \\theta_1|

        Arguments:
            theta1 (float): Start parameter angle in radians.
            theta2 (float): End parameter angle in radians.

        Returns:
            float: Area of the sector. Always non-negative.
        """
        return self.a * self.b * np.abs(theta2 - theta1) / 2

    @staticmethod
    def from_points(points):
        """Fit an ellipse to a set of 2D points using least squares.

        Fixes ``F = 1`` and solves for ``[A, B, C, D, E]`` via least squares on
        the linearised conic equation :math:`Ax^2 + Bxy + Cy^2 + Dx + Ey = -1`.

        Arguments:
            points (array-like): An ``(N, 2)`` array of 2D points. At least 6
                points are required for an over-determined system.

        Returns:
            Ellipse: The fitted ellipse.

        Raises:
            AssertionError: If ``points`` is not an ``(N, 2)`` array or fewer
                than 6 points are provided.
            AssertionError: If the fitted conic is not an ellipse
                (:math:`B^2 - 4AC \\geq 0`).
        """
        points = np.array(points)
        assert points.ndim == 2 and points.shape[1] == 2, 'points must be (N, 2)'
        assert len(points) >= 6, 'At least 6 points required'
        x, y = points[:, 0], points[:, 1]
        Phi = np.column_stack([x**2, x * y, y**2, x, y])
        b = -np.ones(len(points))
        result, _, _, _ = np.linalg.lstsq(Phi, b, rcond=None)
        A, B, C, D, E = result
        return Ellipse(A, B, C, D, E, 1.0)

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
        results = np.sum((points @ self.M) * points, axis=1) + points @ np.array([self.D, self.E]) + self.F
        if points.shape[0] == 1:
            return results.item()
        return results

    def parametric(self, theta):
        """Point(s) on the ellipse boundary at parameter angle(s) ``theta``.

        Uses the standard parametric form :math:`(a\\cos\\theta, b\\sin\\theta)`
        in the ellipse's local frame, transformed to the global frame via ``T``.

        Arguments:
            theta (float or array-like): Parameter angle(s) in radians.

        Returns:
            numpy.ndarray: A ``(2,)`` point for scalar ``theta``, or an
            ``(N, 2)`` array of points for array-like ``theta``, in the
            global frame.
        """
        scalar_input = np.ndim(theta) == 0
        theta = np.atleast_1d(theta).astype(float)
        local = np.vstack([self.a * np.cos(theta), self.b * np.sin(theta), np.ones_like(theta)])
        points = (self.T @ local)[:2].T
        return points[0] if scalar_input else points

    def closest_point(self, point):
        """Find the closest point on the ellipse boundary to a point.

        The closest point is the one at which the ellipse's normal line
        passes through ``point``.

        Arguments:
            point (array-like): A single 2D point ``[x, y]`` in the global frame.

        Returns:
            numpy.ndarray: The closest point on the ellipse, in the global
            frame. If ``point`` already lies on the boundary, it is returned
            unchanged.

        Raises:
            AssertionError: If more than one point is passed.

        Warns:
            UserWarning: If ``point`` coincides with the center of a circle
                (``a == b``); every boundary point is then equally close, and
                the point on the +X axis is returned as an arbitrary but
                valid answer.
        """
        assert np.atleast_2d(point).shape[0] == 1
        point = np.asarray(point, dtype=float)
        if np.isclose(self.evaluate(point), 0, atol=1e-5):
            return point
        X1, Y1 = (self.T_inv @ [point[0], point[1], 1])[:2]
        if np.isclose(self.a, self.b, atol=1e-5) and np.allclose([X1, Y1], 0, atol=1e-5):
            warnings.warn(
                'point coincides with the center of a circle; every point on the '
                'boundary is equally close, returning the point on the +X axis'
            )
            return self.parametric(0)

        c2 = self.a**2 - self.b**2
        if np.isclose(Y1, 0, atol=1e-5):
            thetas = [0, np.pi]
            if not np.isclose(c2, 0, atol=1e-5):
                cos_value = X1 * self.a / c2
                if abs(cos_value) <= 1:
                    theta = np.arccos(cos_value)
                    thetas.extend([theta, -theta])
        else:
            coeffs = [
                Y1 * self.b,
                2 * (X1 * self.a + c2),
                0.0,
                2 * (X1 * self.a - c2),
                -Y1 * self.b
            ]

            all_roots = np.roots(coeffs)
            real_roots_mask = np.abs(all_roots.imag) < 1e-5
            real_roots = all_roots[real_roots_mask].real

            thetas = np.arctan2(2 * real_roots, 1 - real_roots**2)

        eps = self.parametric(thetas)  # global
        dists = np.linalg.norm(eps - point, axis=1)
        return eps[np.argmin(dists)]
