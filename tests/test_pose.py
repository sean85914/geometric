import pytest
import numpy as np
from geometric.pose import Pose


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
