import pytest
import numpy as np
from geometric.pose import Pose, _skew, _left_jacobian, _left_jacobian_inv


# ── Skew ──────────────────────────────────────────────────────────────────────

class TestSkew:
    def test_antisymmetric(self):
        W = _skew([1, 2, 3])
        assert np.allclose(W + W.T, 0)

    def test_values(self):
        W = _skew([1, 2, 3])
        expected = [
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ]
        assert np.allclose(W, expected)


# ── Left Jacobian and Inverse ─────────────────────────────────────────────────

class TestLeftJacobian:
    def test_inserse(self):
        omega = [0.1, 0.2, 0.3]
        assert np.allclose(_left_jacobian(omega) @ _left_jacobian_inv(omega), np.eye(3))

    def test_near_zero(self):
        omega = [1e-6, 0, 0]
        assert np.allclose(_left_jacobian(omega), np.eye(3), atol=1e-4)
        assert np.allclose(_left_jacobian_inv(omega), np.eye(3), atol=1e-4)


# ── Construction ──────────────────────────────────────────────────────────────

class TestPoseConstruction:
    def test_identity(self):
        p = Pose.identity()
        assert np.allclose(p.matrix, np.eye(4))

    def test_translation_only(self):
        p = Pose([1, 2, 3], [0, 0, 0])
        assert np.allclose(p.translation, [1, 2, 3])
        assert np.allclose(p.matrix[:3, :3], np.eye(3))

    def test_from_matrix_roundtrip(self):
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        p = Pose.from_matrix(T)
        assert np.allclose(p.matrix, T)

    def test_invalid_trans_raises(self):
        with pytest.raises(AssertionError):
            Pose([1, 2], [0, 0, 0])

    def test_invalid_rot_raises(self):
        with pytest.raises(AssertionError):
            Pose([1, 2, 3], [0, 0])


# ── Set rotation ──────────────────────────────────────────────────────────────

class TestSetRotation:
    def test_from_rotvec(self):
        rv = [0, 0, np.pi / 2]
        p = Pose.identity().set_rotation_from_rotvec(rv)
        # x-axis should map to y-axis
        x_transformed = p.matrix[:3, :3] @ [1, 0, 0]
        assert np.allclose(x_transformed, [0, 1, 0], atol=1e-6)

    def test_from_matrix(self):
        rot = np.eye(3)
        p = Pose.identity().set_rotation_from_matrix(rot)
        assert np.allclose(p.matrix[:3, :3], np.eye(3))

    def test_from_quaternion(self):
        # 90° around z: quaternion [0, 0, sin(pi/4), cos(pi/4)]
        q = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
        p = Pose.identity().set_rotation_from_quaternion(q)
        x_transformed = p.matrix[:3, :3] @ [1, 0, 0]
        assert np.allclose(x_transformed, [0, 1, 0], atol=1e-6)

    def test_from_axis_angle(self):
        p = Pose.identity().set_rotation_from_axis_angle([0, 0, 1], float(np.pi / 2))
        x_transformed = p.matrix[:3, :3] @ [1, 0, 0]
        assert np.allclose(x_transformed, [0, 1, 0], atol=1e-6)

    def test_from_euler(self):
        p = Pose.identity().set_rotation_from_euler([0, 0, np.pi / 2], 'xyz')
        x_transformed = p.matrix[:3, :3] @ [1, 0, 0]
        assert np.allclose(x_transformed, [0, 1, 0], atol=1e-6)


# ── get_* representations ─────────────────────────────────────────────────────

class TestGetRepresentations:
    def _pose_90z(self):
        return Pose.identity().set_rotation_from_axis_angle([0, 0, 1], float(np.pi / 2))

    def test_get_rotvec_matches_rotation(self):
        rv = [0, 0, np.pi / 3]
        p = Pose.identity().set_rotation_from_rotvec(rv)
        assert np.allclose(p.get_rotvec(), rv, atol=1e-6)

    def test_get_quaternion_unit_length(self):
        p = self._pose_90z()
        q = p.get_quaternion()
        assert np.isclose(np.linalg.norm(q), 1.0)

    def test_get_euler_xyz(self):
        angles = [0.1, 0.2, 0.3]
        p = Pose.identity().set_rotation_from_euler(angles, 'xyz')
        recovered = p.get_euler('xyz')
        assert np.allclose(recovered, angles, atol=1e-6)

    def test_get_axis_angle_identity(self):
        p = Pose.identity()
        axis, angle = p.get_axis_angle()
        assert np.isclose(angle, 0.0)

    def test_get_axis_angle_90z(self):
        p = self._pose_90z()
        axis, angle = p.get_axis_angle()
        assert np.isclose(angle, np.pi / 2, atol=1e-6)
        assert np.allclose(np.abs(axis), [0, 0, 1], atol=1e-6)


# ── Multiplication and inverse ────────────────────────────────────────────────

class TestCompose:
    def test_identity_composition(self):
        p = Pose([1, 2, 3], [0.1, 0.2, 0.3])
        assert p @ Pose.identity() == p
        assert Pose.identity() @ p == p

    def test_inverse(self):
        p = Pose([1, 2, 3], [0.1, 0.2, 0.3])
        result = p @ p.inv
        assert result == Pose.identity()

    def test_chain(self):
        p1 = Pose([1, 0, 0], [0, 0, 0])
        p2 = Pose([0, 1, 0], [0, 0, 0])
        chained = Pose.chain(p1, p2)
        expected = p1 @ p2
        assert chained == expected

    def test_diff(self):
        p1 = Pose([1, 0, 0], [0, 0, 0])
        p2 = Pose([3, 0, 0], [0, 0, 0])
        d = p1.diff(p2)
        # p1 @ d == p2
        assert (p1 @ d) == p2

    def test_diffs(self):
        base = Pose([1, 0, 0], [0, 0, 0])
        targets = [Pose([2, 0, 0], [0, 0, 0]), Pose([3, 0, 0], [0, 0, 0])]
        diffs = base.diffs(targets)
        for d, t in zip(diffs, targets):
            assert (base @ d) == t


# ── Transform points ──────────────────────────────────────────────────────────

class TestTransformPoints:
    def test_translation_only(self):
        p = Pose([1, 2, 3], [0, 0, 0])
        result = p.transform_point([0, 0, 0])
        assert np.allclose(result, [1, 2, 3])

    def test_rotation_90z(self):
        p = Pose.identity().set_rotation_from_axis_angle([0, 0, 1], float(np.pi / 2))
        result = p.transform_point([1, 0, 0])
        assert np.allclose(result, [0, 1, 0], atol=1e-6)

    def test_transform_multiple_points(self):
        p = Pose([1, 0, 0], [0, 0, 0])
        pts = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        results = p.transform_points(pts)
        assert np.allclose(results, [[1, 0, 0], [2, 0, 0], [3, 0, 0]])

    def test_rotate_point(self):
        center = [1, 0, 0]
        p = Pose(center, [0, 0, 0]).set_rotation_from_axis_angle([0, 0, 1], float(np.pi / 2))
        # Point at (2, 0, 0) rotated 90° around z through center (1,0,0) → (1, 1, 0)
        result = p.rotate_point([2, 0, 0])
        assert np.allclose(result, [1, 1, 0], atol=1e-6)


# ── Interpolation ─────────────────────────────────────────────────────────────

class TestInterpolate:
    def test_translation_linear(self):
        p1 = Pose([0, 0, 0], [0, 0, 0])
        p2 = Pose([10, 0, 0], [0, 0, 0])
        poses = p1.interpolate(p2, 11)
        xs = [p.translation[0] for p in poses]
        assert np.allclose(xs, list(range(11)))

    def test_endpoints(self):
        p1 = Pose([0, 0, 0], [0, 0, 0])
        p2 = Pose([1, 2, 3], [0.1, 0.2, 0.3])
        poses = p1.interpolate(p2, 5)
        assert np.allclose(poses[0].translation, p1.translation)
        assert np.allclose(poses[-1].translation, p2.translation, atol=1e-6)

    def test_num_less_than_3_raises(self):
        p1 = Pose.identity()
        p2 = Pose([1, 0, 0], [0, 0, 0])
        with pytest.raises(AssertionError):
            p1.interpolate(p2, 2)


# ── Distance ──────────────────────────────────────────────────────────────────

class TestDistance:
    def test_from_origin(self):
        p = Pose([3, 4, 0], [0, 0, 0])
        assert np.isclose(p.distance(), 5.0)

    def test_from_ref(self):
        p = Pose([3, 4, 0], [0, 0, 0])
        assert np.isclose(p.distance([0, 4, 0]), 3.0)


# ── From Twist and To Twist ───────────────────────────────────────────────────

class TestFromToTwist:
    def test_roundtrip(self):
        # norm(omega) should less than pi
        twist = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
        assert np.allclose(Pose.from_twist(twist).to_twist(), twist)

    def test_zero_twist(self):
        assert Pose.from_twist([0] * 6) == Pose.identity()

    def test_pure_rotation(self):
        p = Pose.from_twist([0, 0, np.pi / 2, 0, 0, 0])
        assert np.allclose(p.translation, [0, 0, 0])

    def test_pure_translation(self):
        p = Pose.from_twist([0, 0, 0, 1, 2, 3])
        assert np.allclose(p.translation, [1, 2, 3])
        assert np.allclose(p.matrix[:3, :3], np.eye(3))

    def test_invalid_length(self):
        with pytest.raises(AssertionError):
            Pose.from_twist([1, 2, 3])


# ──  Geodesic ─────────────────────────────────────────────────────────────────

class TestTwistToGeodesic:
    def test_twist_to_self(self):
        p = Pose.random_pose()
        assert np.allclose(p.twist_to(p), 0)

    def test_geodesic_distance_self(self):
        p = Pose.random_pose()
        assert np.isclose(p.geodesic_distance(p), 0)

    def test_geodesic_distance_positive(self):
        p1 = Pose.identity()
        p2 = Pose([1, 0, 0], [0, 0, 0])
        assert p1.geodesic_distance(p2) > 0

    def test_twist_to_recovers_pose(self):
        p1 = Pose.random_pose()
        p2 = Pose.random_pose()
        recovered = p1 @ Pose.from_twist(p1.twist_to(p2))
        assert recovered == p2


# ──  Mean ─────────────────────────────────────────────────────────────────────

class TestMean:
    def test_single_pose(self):
        p = Pose.random_pose()
        assert Pose.mean([p]) == p

    def test_identical_poses(self):
        p = Pose.random_pose()
        assert Pose.mean([p, p, p]) == p

    def test_mean_translation(self):
        p1 = Pose([0, 0, 0], [0, 0, 0])
        p2 = Pose([2, 0, 0], [0, 0, 0])
        assert np.allclose(Pose.mean([p1, p2]).translation, [1, 0, 0])

    def test_mean_rotation(self):
        p1 = Pose([0, 0, 0], [0, 0, 0])
        p2 = Pose([0, 0, 0], [0, 0, np.pi / 2])
        pm = Pose.mean([p1, p2])
        assert np.isclose(pm.get_euler('xyz')[2], np.pi / 4, atol=1e-5)

    def test_weights(self):
        p1 = Pose([0, 0, 0], [0, 0, 0])
        p2 = Pose([4, 0, 0], [0, 0, 0])
        pm = Pose.mean([p1, p2], weights=[1, 3])
        assert np.allclose(pm.translation, [3, 0, 0])

    def test_empty_raises(self):
        with pytest.raises(AssertionError):
            Pose.mean([])

    def test_weights_mismatch_raises(self):
        with pytest.raises(AssertionError):
            Pose.mean([Pose.identity(), Pose.identity()], weights=[1])
