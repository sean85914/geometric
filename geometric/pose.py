import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from .geometric import unit_vector, angle_between_vectors, norm


class Pose:
    """Represents a 3D pose composed of a translation and a rotation, stored as a 4x4 transformation matrix.

    Attributes:
        matrix (nump.ndarray): The 4x4 transformation matrix.
        translation (nump.ndarray): The 3D translation vector.
        rotation (numpy.ndarray): The rotation represented as a rotation vector.
        inv (Pose): The inverse of the pose.
    """

    def __init__(self, trans, rot):
        """Initializes a Pose object.

        Arguments:
            trans (array-like): A 3-element translation vector.
            rot (array-like): A 3-element rotation vector (Rodrigues).

        Raises:
            AssertionError: If input vectors are not of length 3.
        """
        assert len(trans) == len(rot) == 3
        self._T = np.eye(4)
        self._T[:3, :3] = R.from_rotvec(rot).as_matrix()
        self._T[:3, 3] = np.array(trans)
        self._translation = np.array(trans)
        self._rotation = np.array(rot)

    def __repr__(self):
        """Returns the string representation of the transformation matrix.

        Returns:
            str: The transformation matrix as a string.
        """
        return str(self._T)

    def __str__(self):
        """Returns a user-friendly string of the pose's translation and rotation.

        Returns:
            str: The translation and rotation as a string.
        """
        return f'Translation: {self.translation}, rotation: {self.rotation}'

    def __eq__(self, rhs):
        """Checks if two Pose objects are approximately equal.

        Arguments:
            rhs (Pose): Another Pose instance.

        Returns:
            bool: True if matrices are approximately equal, False otherwise.
        """
        assert isinstance(rhs, Pose)
        return np.allclose(self.matrix, rhs.matrix)

    @property
    def matrix(self):
        return self._T

    @property
    def translation(self):
        return self._translation

    def set_translation(self, trans):
        """Sets the translation component of the pose.

        Arguments:
            trans (array-like): A 3-element translation vector.

        Returns:
            Pose: The modified pose object (`self`), allowing for method chaining.

        Raises:
            AssertionError: If input vector is not of length 3.
        """
        assert len(trans) == 3
        self._translation = trans
        self._T[:3, 3] = trans
        return self

    @property
    def rotation(self):
        return self._rotation

    def set_rotation_from_matrix(self, rot):
        """Sets the rotation from a 3x3 rotation matrix.

        Arguments:
            rot (array-like): A 3x3 rotation matrix.

        Returns:
            Pose: The modified pose object (`self`), allowing for method chaining.

        Raises:
            AssertionError: If input is not a valid 3x3 matrix.
        """
        assert np.array(rot).shape == (3, 3)
        r = R.from_matrix(rot)
        self._rotation = r.as_rotvec()
        self._T[:3, :3] = r.as_matrix()
        return self

    def set_rotation_from_rotvec(self, rv):
        """Sets the rotation from a rotation vector.

        Args:
            rv (array-like): A 3-element rotation vector.

        Returns:
            Pose: The modified pose object (`self`), allowing for method chaining.

        Raises:
            AssertionError: If input vector is not of length 3.
        """
        assert len(rv) == 3
        r = R.from_rotvec(rv)
        self._rotation = rv
        self._T[:3, :3] = r.as_matrix()
        return self

    def set_rotation_from_quaternion(self, quat):
        """Sets the rotation from a quaternion.

        Arguments:
            quat (array-like): A 4-element quaternion ``[x, y, z, w]``.

        Returns:
            Pose: The modified pose object (`self`), allowing for method chaining.

        Raises:
            AssertionError: If input quaternion is not of length 4.
        """
        assert len(quat) == 4
        r = R.from_quat(quat)
        self._rotation = r.as_rotvec()
        self._T[:3, :3] = r.as_matrix()
        return self

    def set_rotation_from_axis_angle(self, axis, angle):
        """
        Sets the rotation from an axis and angle.

        Arguments:
            axis (array-like): A 3-element rotation axis.
            angle (float): The rotation angle in radians.

        Returns:
            Pose: The modified pose object (`self`), allowing for method chaining.

        Raises:
            AssertionError: If axis is not of length 3 or angle is not a float.
        """
        assert len(axis) == 3 and isinstance(angle, float)
        axis = np.array(axis)
        unit_axis = axis / np.linalg.norm(axis)
        rv = unit_axis * angle
        self.set_rotation_from_rotvec(rv)
        return self

    def set_rotation_from_euler(self, angles, sequence):
        """Sets the rotation from Euler angles.

        Args:
            angles (array-like): A 3-element list of Euler angles.
            sequence (str): A valid 3-character rotation sequence (e.g., ``xyz``).

        Returns:
            Pose: The modified pose object (`self`), allowing for method chaining.

        Raises:
            AssertionError: If input is invalid.

        See Also:
            See sequence definition from `scipy <https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.Rotation.from_euler>`_
        """
        assert len(angles) == 3
        assert re.search('([x-zX-Z]){3}', sequence) and (sequence.islower() or sequence.isupper())
        r = R.from_euler(sequence, angles)
        self._rotation = r.as_rotvec()
        self._T[:3, :3] = r.as_matrix()
        return self

    def __matmul__(self, rhs):
        """Applies transformation composition using matrix multiplication.

        Arguments:
            rhs (Pose): Another Pose instance.

        Returns:
            Pose: A new Pose resulting from self * rhs.
        """
        assert isinstance(rhs, Pose)
        return Pose.from_matrix(self._T @ rhs._T)

    @staticmethod
    def chain(*poses):
        '''Chains multiple Pose instances via left-to-right multiplication.

        Arguments:
            *poses (Pose): Any number of Pose instances.

        Returns:
            Pose: The resulting Pose from multiplying all inputs.

        Raises:
            AssertionError: If any input is not a Pose.
        '''
        assert all(isinstance(p, Pose) for p in poses), "All inputs must be Pose instances"
        if not poses:
            return Pose.identity()
        T = np.eye(4)
        for p in poses:
            T = T @ p.matrix
        return Pose.from_matrix(T)

    @classmethod
    def from_matrix(cls, matrix):
        """Creates a Pose instance from a 4x4 transformation matrix.

        Arguments:
            matrix (np.ndarray): A 4x4 homogeneous transformation matrix.

        Returns:
            Pose: A new Pose instance.

        Raises:
            AssertionError: If matrix is not valid.
        """
        assert isinstance(matrix, np.ndarray) \
               and matrix.shape == (4, 4) \
               and np.isclose(matrix[3, 3], 1.0, atol=1e-5)
        trans = matrix[:3, 3]
        rv = R.from_matrix(matrix[:3, :3]).as_rotvec()
        return cls(trans, rv)

    @staticmethod
    def identity():
        """Returns the identity Pose.

        Returns:
            Pose: A Pose representing the identity transformation.
        """
        return Pose.from_matrix(np.eye(4))

    @staticmethod
    def random_pose():
        """
        Generates a random Pose with translation in [0, 1] and rotation up to :math:`\\pi` radians.

        Returns:
            Pose: A randomly generated Pose.
        """
        trans = np.random.rand(3)
        rot = (np.random.rand(3)) * np.random.uniform(0, np.pi)
        return Pose(trans, rot)

    @staticmethod
    def from_ros_geometry_pose(pose):
        """Converts a ROS `geometry_msgs/Pose <https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Pose.html>`_
        message to a Pose.

        Arguments:
            pose: A geometry_msgs/Pose object.

        Returns:
            Pose: A new Pose instance.
        """

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
        """Converts a ROS `geometry_msgs/PoseStamped <https://docs.ros.org/en/noetic/api/geometry_msgs/
        html/msg/PoseStamped.html>`_ message to a Pose.

        Arguments:
            pose: A geometry_msgs/PoseStamped object.

        Returns:
            Pose: A new Pose instance.
        """
        assert type(pose).__name__ == 'PoseStamped'
        return Pose.from_ros_geometry_pose(pose.pose)

    @property
    def inv(self):
        return Pose.from_matrix(np.linalg.inv(self._T))

    def diff(self, rhs):
        """Computes the difference transformation between self and rhs.

        Arguments:
            rhs (Pose): The target pose.

        Returns:
            Pose: The relative transformation Pose

        Note:
            ``self * diff = rhs``
        """
        '''this * diff = rhs
        -> diff = this.inv @ rhs
        '''
        assert isinstance(rhs, Pose)
        return Pose.from_matrix((self.inv @ rhs).matrix)

    def interpolate(self, target_pose, num):
        """Interpolates between self and a target pose.

        Arguments:
            target_pose (Pose): The target pose.
            num (int): Number of interpolation steps, **including** the start and end poses. Must greater than 2.

        Returns:
            list of Pose: A list of interpolated Pose instances.

        Raises:
            AssertionError: If `target_pose` is not a Pose.
            AssertionError: If `num` less than 3.

        The translation is interpolated linearly, while the rotation is interpolated using spherical
        linear interpolation (SLERP).

        .. code-block:: python

            >>> from geometric import Pose
            >>> import numpy as np
            >>> p1 = Pose.identity()
            >>> p2 = Pose([10, 0, 0], [1/np.sqrt(3) * np.radians(120)] * 3)
            >>> ps = p1.interpolate(p2, 11)
            >>> print([p.translation[0] for p in ps])
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        Visualization with `Open3d <https://www.open3d.org/docs/release/>`_

        .. code-block:: python

            >>> import open3d as o3d
            >>> geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(0.5).transform(p.matrix) for p in ps]
            >>> o3d.visualization.draw_geometries(geoms)

        Result

        .. figure:: _static/interpolate.png
            :align: center
        """
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

    def set_rotation_from_vector(self, vector, target_axis=2):
        '''Set the rotation such that the reference axis is rotated to align with the given vector.

        Arguments:
            vector (list or array-like): The target 3D direction vector to align with.
            target_axis (int, optional): The index of the original axis to align (0: X, 1: Y, 2: Z). Defaults to 2.

        Returns:
            Pose: The modified pose object (`self`), allowing for method chaining.

        Raises:
            AssertionError: If the axis index is not between 0 and 2 (inclusive).
        '''
        assert target_axis >= 0 and target_axis <= 2
        origin = np.array([0.0, 0.0, 0.0])
        origin[target_axis] = 1.0
        if np.allclose(origin, vector):
            # arbitrary axis
            axis = vector
            angle = 0.0
        elif np.allclose(-origin, vector):
            axis = np.array([0.0, 0.0, 0.0])
            axis[(target_axis + 1) % 3] = 1.0
            angle = np.pi
        else:
            axis = np.cross(origin, vector)
            axis = unit_vector(axis)
            angle = angle_between_vectors(vector, origin)
        self.set_rotation_from_axis_angle(axis, angle)
        return self

    def distance(self, ref_point=np.zeros(3)):
        '''Compute the Euclidean distance from this pose's translation to a given reference point.

        Arguments:
            ref_point (array-like, optional): A 3D reference point. Defaults to ``[0, 0, 0]``.

        Returns:
            float: The distance from the `ref_point` to this pose's translation.

        Raises:
            AssertionError: If the point is not with length 3
        '''
        assert len(ref_point) == 3
        return np.linalg.norm(self.translation - ref_point)

    def get_rotvec(self):
        '''Get the rotation vector (Rodrigues vector) representation of the pose's orientation.

        Returns:
            numpy.ndarray: A 3-element rotation vector.
        '''
        return R.from_matrix(self.matrix[:3, :3]).as_rotvec()

    def get_quaternion(self):
        '''Get the quaternion representation of the pose's orientation.

        Returns:
            numpy.ndarray: A 4-element quaternion in ``[x, y, z, w]`` format.
        '''
        return R.from_matrix(self.matrix[:3, :3]).as_quat()

    def get_euler(self, sequence):
        '''Get the Euler angles of the pose's orientation in the specified rotation sequence.

        Arguments:
            sequence (str): A 3-character string indicating the rotation axes order (e.g., ``xyz``).

        Returns:
            numpy.ndarray: A 3-element array of Euler angles (in radians).

        Raises:
            AssertionError: If the sequence format is invalid.

        See Also:
            See sequence definition from `scipy <https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.Rotation.from_euler>`_
        '''
        assert re.search('([x-zX-Z]){3}', sequence) and (sequence.islower() or sequence.isupper())
        return R.from_matrix(self.matrix[:3, :3]).as_euler(sequence)

    def get_axis_angle(self):
        '''Get the axis-angle representation of the pose's orientation.

        Returns:
            tuple: A tuple (axis, angle), where:

                - axis (numpy.ndarray): A 3D unit vector representing the rotation axis.
                - angle (float): Rotation angle in radians.
        '''
        rv = self.get_rotvec()
        angle = norm(rv)
        axis = unit_vector(rv)
        return axis, angle

    def transform_point(self, point):
        '''Apply the pose transformation to a single 3D point.

        Arguments:
            point (array-like): A 3D point to transform.

        Returns:
            numpy.ndarray: The transformed 3D point.

        Raises:
            AssertionError: If the point is not with length 3
        '''
        assert len(point) == 3
        points = np.array(point).reshape(1, 3)
        return self.transform_points(points)[0]

    def transform_points(self, points):
        '''Apply the pose transformation to a list or array of 3D points.

        Arguments:
            points (array-like): A ``(N, 3)`` array of 3D points.

        Returns:
            numpy.ndarray: The transformed points, shape ``(N, 3)``.

        Raises:
            AssertionError: If input does not have shape ``(N, 3)``.
        '''
        points = np.array(points)
        assert points.shape[1] == 3
        return (self.matrix[:3, :3] @ points.T).T + self.matrix[:3, 3]

    def rotate_point(self, point):
        '''Rotate a single 3D point around the pose's translation.

        This applies the rotation component of the pose to the point,
        treating `self.translation` as the rotation center.

        Args:
            point (array-like): A 3-element 3D point.

        Returns:
            numpy.ndarray: The rotated 3D point.

        Raises:
            AssertionError: If the input does not have length 3.
        '''
        assert len(point) == 3
        points = np.array(point).reshape(1, 3)
        return self.rotate_points(points)[0]


    def rotate_points(self, points):
        '''Rotate multiple 3D points around the pose's translation.

        This function rotates each point around `self.translation` using
        the pose's rotation matrix, without applying any translation.

        Args:
            points (array-like): A (N, 3) array of 3D points.

        Returns:
            numpy.ndarray: A (N, 3) array of rotated 3D points.

        Raises:
            AssertionError: If input does not have shape (N, 3).
        '''
        points = np.array(points)
        assert points.shape[1] == 3
        p_trans = Pose.identity().set_translation(self.translation)
        p_rot = Pose.identity().set_rotation_from_matrix(self.rotation)
        p = p_trans @ p_rot @ p_trans.inv
        return p.transform_points(points)