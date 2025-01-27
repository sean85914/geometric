from enum import Enum


class PointTriangleEnum(Enum):
    INSIDE = 0
    ON_BORDER = 1
    OUTSIDE = 2
    NOT_IN_PLANE = 3  # 3D


class PointCircleEnum(Enum):
    INSIDE = 0
    ON_BORDER = 1
    OUTSIDE = 2
    NOT_IN_PLANE = 3  # 3D


class PointShapeEnum(Enum):
    INSIDE = 0
    ON_BORDER = 1
    OUTSIDE = 2
