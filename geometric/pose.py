import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class Pose:
    def __init__(self, trans, rot):
        assert len(trans) == len(rot) == 3
        self._T = np.eye(4)
        self._T[:3, :3] = R.from_rotvec(rot).as_matrix()
        self._T[:3, 3] = trans
        self._translation = trans
        self._rotation = rot

    def __repr__(self):
        return str(self._T)

    def __str__(self):
        return f'Translation: {self.translation}, rotation: {self.rotation}'

    @property
    def matrix(self):
        return self._T

    @property
    def translation(self):
        return self._translation

    def set_translation(self, trans):
        assert len(trans) == 3
        self._translation = self._translation
        self._T[:3, 3] = trans

    @property
    def rotation(self):
        return self._rotation

    def set_rotation_from_matrix(self, rot):
        assert np.array(rot).shape == (3, 3)
        self._rotation = R.from_matrix(rot).as_matrix()
        self._T[:3, :3] = self._rotation

    def set_rotation_from_rotvec(self, rv):
        assert len(rv) == 3
        self._rotation = R.from_rotvec(rv).as_matrix()
        self._T[:3, :3] = self._rotation

    def set_rotation_from_quaternion(self, quat):
        assert len(quat) == 4
        self._rotation = R.from_quat(quat).as_matrix()
        self._T[:3, :3] = self._rotation

    def set_rotation_from_axis_angle(self, axis, angle):
        assert len(axis) == 3 and isinstance(angle, float)
        axis = np.array(axis)
        unit_axis = axis / np.linalg.norm(axis)
        rv = unit_axis * angle
        self.set_rotation_from_rotvec(rv)

    def set_rotation_from_euler(self, angles, sequence):
        assert len(angles) == 3
        assert re.search('([x-zX-Z]){3}', sequence) and (sequence.islower() or sequence.isupper())
        self._rotation = R.from_euler(sequence, angles).as_matrix()
        self._T[:3, :3] = self._rotation

    def __matmul__(self, rhs):
        assert isinstance(rhs, Pose)
        return Pose.from_matrix(self._T @ rhs._T)

    @classmethod
    def from_matrix(cls, matrix):
        assert isinstance(matrix, np.ndarray) \
               and matrix.shape == (4, 4) \
               and np.isclose(matrix[3, 3], 1.0, atol=1e-5)
        trans = matrix[:3, 3]
        rv = R.from_matrix(matrix[:3, :3]).as_rotvec()
        return cls(trans, rv)

    @staticmethod
    def identity():
        return Pose.from_matrix(np.eye(4))

    @staticmethod
    def random_pose():
        trans = np.random.rand(3)
        rot = (np.random.rand(3)) * np.random.uniform(0, np.pi)
        return Pose(trans, rot)

    @staticmethod
    def from_ros_geometry_pose(pose):
        assert type(pose).__name__ == 'Pose'
        trans = []
        for axis in ['x', 'y', 'z']:
            trans.append(getattr(pose.position, axis))
        quat = []
        for axis in ['x', 'y', 'z', 'w']:
            quat.append(getattr(pose.orientation, axis))
        rv = R.from_quaternion(quat).as_rotvec()
        return Pose(trans, rv)

    @staticmethod
    def from_ros_geometry_pose_stamped(pose):
        assert type(pose).__name__ == 'PoseStamped'
        return Pose.from_ros_geometry_pose(pose.pose)

    @property
    def inv(self):
        return Pose.from_matrix(np.linalg.inv(self._T))

    def diff(self, rhs):
        '''this * diff = rhs
        -> diff = this.inv @ rhs
        '''
        assert isinstance(rhs, Pose)
        return Pose.from_matrix(self.inv @ rhs)

    def interpolate(self, target_pose, num):
        '''  0    ...      num - 1
           this          target_pose
        '''
        assert isinstance(target_pose, Pose)
        assert isinstance(num, int) and num > 2
        poses = []
        trans_diff = target_pose.translation - self.translation
        slerp = Slerp([0, num - 1], R.from_rotvec([self.rotation, target_pose.rotation]))
        slerp_res = slerp(list(range(num)))
        for i in range(num):
            trans = self.translation + i / (num - 1) * trans_diff
            rot = slerp_res[i].as_rotvec()
            poses.append(Pose(trans, rot))
        return poses
