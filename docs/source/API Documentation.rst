API Documentation
==================

Vectors and Points
~~~~~~~~~~~~~~~~~~
.. autofunction:: geometric.norm
.. autofunction:: geometric.is_zero_vector
.. autofunction:: geometric.is_on_axis
.. autofunction:: geometric.middle_point
.. autofunction:: geometric.average_point
.. autofunction:: geometric.distance_between_points
.. autofunction:: geometric.angle_between_vectors
.. autofunction:: geometric.orthogonal_vector

Lines
~~~~~
.. autofunction:: geometric.line_from_point_vector
.. autofunction:: geometric.line_from_two_points
.. autofunction:: geometric.random_point_on_line
.. autofunction:: geometric.is_point_on_line

Planes
~~~~~~
.. autofunction:: geometric.plane_from_three_points
.. autofunction:: geometric.plane_from_point_vector
.. autofunction:: geometric.random_point_on_plane
.. autofunction:: geometric.is_point_on_plane
.. autofunction:: geometric.is_line_on_plane

Bisector
~~~~~~~~~
.. autofunction:: geometric.perpendicular_bisector
.. autofunction:: geometric.angle_bisector_line_from_two_lines
.. autofunction:: geometric.angle_bisector_plane_from_two_lines
.. autofunction:: geometric.angle_bisector_plane_from_two_planes

Projection and Distance
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: geometric.nearest_point
.. autofunction:: geometric.nearest_distance
.. autofunction:: geometric.vector_projection
.. autofunction:: geometric.project_vector_on_plane
.. autofunction:: geometric.project_point_on_line
.. autofunction:: geometric.project_point_on_plane
.. autofunction:: geometric.project_line_on_plane
.. autofunction:: geometric.distance_point_to_line
.. autofunction:: geometric.distance_point_to_plane

Intersection
~~~~~~~~~~~~
.. autofunction:: geometric.intersection_between_lines
.. autofunction:: geometric.intersection_between_line_segments
.. autofunction:: geometric.intersection_between_line_and_circle
.. autofunction:: geometric.line_from_planes
.. autofunction:: geometric.point_from_plane_and_line
.. autofunction:: geometric.point_from_three_planes

Circle
~~~~~~
.. autofunction:: geometric.circle_from_three_points
.. autofunction:: geometric.circle_from_center_and_points
.. autofunction:: geometric.generate_points_on_circle
.. autofunction:: geometric.circle_coordinate_transform

Arc
~~~
.. autofunction:: geometric.arc_from_center_and_endpoints
.. autofunction:: geometric.arc_from_three_points
.. autofunction:: geometric.generate_points_on_arc
.. autofunction:: geometric.is_point_on_arc

Sphere
~~~~~~
.. autofunction:: geometric.distance_between_points_on_sphere
.. autofunction:: geometric.random_point_on_sphere
.. autofunction:: geometric.spherical_cap_volume
.. autofunction:: geometric.overlap_volume_between_spheres

Parameters from Noisy Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: geometric.line_from_noisy_data
.. autofunction:: geometric.plane_from_noisy_data
.. autofunction:: geometric.circle_from_noisy_data


Relation between Point and Shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: geometric.point_triangle_relation
.. autofunction:: geometric.point_cylinder_relation
.. autofunction:: geometric.point_sphere_relation
.. autofunction:: geometric.point_circle_relation

Transform between Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: geometric.cartesian_to_spherical
.. autofunction:: geometric.spherical_to_cartesian

Pose
~~~~
.. autoclass:: geometric.Pose
    :members: