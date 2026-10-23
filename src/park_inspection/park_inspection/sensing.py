"""Depth-camera point transforms for the PX4 x500_depth model.

Gazebo Harmonic point clouds follow the sensor frame (+X forward, +Y left,
+Z up), which is also the convention the gz-sensors maintainers document for
depth-camera clouds. Image topics use a separate optical frame. The OakD-Lite
mount on ``x500_depth`` has no rotation relative to ``base_link``, so a
forward-frame cloud only needs the sensor origin in the body.
"""

import math
import struct
from typing import Iterable, List, Sequence, Tuple

Point3 = Tuple[float, float, float]

# model://x500_depth joint pose plus the StereoOV7251 pose on camera_link.
X500_DEPTH_SENSOR_POSITION_BODY: Point3 = (0.13233, 0.0, 0.26078)
# Top-deck 16-beam lidar origin. The x500 plate is about 0.05 m above base_link.
X500_LIDAR_POSITION_BODY: Point3 = (0.0, 0.0, 0.09)

_FLOAT32 = 7


def rotate_body_to_map(quaternion: Sequence[float], vector: Point3) -> Point3:
    """Rotate a body-frame vector by a geometry_msgs quaternion (x, y, z, w)."""
    qx, qy, qz, qw = (float(value) for value in quaternion)
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def sensor_point_to_body(point: Point3, convention: str) -> Point3:
    """Express one cloud point in the body frame, before adding the mount offset."""
    if convention == "x_forward":
        return point
    if convention == "optical":
        return (point[2], -point[0], -point[1])
    if convention == "map":
        raise ValueError("map-frame clouds are transformed with the vehicle pose")
    raise ValueError(
        "cloud_convention must be 'x_forward', 'optical', or 'map'"
    )


def cloud_to_map_rays(
    points: Iterable[Point3],
    vehicle_position: Point3,
    quaternion: Sequence[float],
    sensor_position_body: Point3 = X500_DEPTH_SENSOR_POSITION_BODY,
    convention: str = "x_forward",
) -> Tuple[Point3, List[Point3]]:
    """Return the sensor origin and cloud endpoints in the map (ENU) frame."""
    origin_offset = rotate_body_to_map(quaternion, sensor_position_body)
    origin = tuple(
        vehicle_position[axis] + origin_offset[axis] for axis in range(3)
    )
    if convention == "map":
        return origin, list(points)

    endpoints: List[Point3] = []
    for point in points:
        body = sensor_point_to_body(point, convention)
        mounted = tuple(
            sensor_position_body[axis] + body[axis] for axis in range(3)
        )
        rotated = rotate_body_to_map(quaternion, mounted)
        endpoints.append(
            tuple(vehicle_position[axis] + rotated[axis] for axis in range(3))
        )
    return origin, endpoints


def iter_laser_scan_points(scan, stride: int = 1) -> Iterable[Point3]:
    """Yield planar lidar returns in the sensor frame (+X forward, +Y left).

    Angles follow ``sensor_msgs/LaserScan``: zero is forward and positive
    angles rotate toward the left. Invalid and maximum-range readings are
    kept at ``range_max`` so the occupancy update can clear free space
    without marking a hit.
    """
    ranges = scan.ranges
    count = len(ranges)
    step = max(1, int(stride))
    angle_min = float(scan.angle_min)
    increment = float(scan.angle_increment)
    range_min = float(scan.range_min)
    range_max = float(scan.range_max)
    for index in range(0, count, step):
        reading = float(ranges[index])
        angle = angle_min + index * increment
        if (
            not math.isfinite(reading)
            or reading >= range_max
            or reading < range_min
        ):
            distance = range_max
        else:
            distance = reading
        yield (
            distance * math.cos(angle),
            distance * math.sin(angle),
            0.0,
        )


def iter_cloud_xyz(cloud, stride: int = 1) -> Iterable[Point3]:
    """Yield finite XYZ points from a ``sensor_msgs/PointCloud2``-like message."""
    offsets = {}
    for field in cloud.fields:
        if field.name in ("x", "y", "z"):
            if field.datatype != _FLOAT32:
                raise ValueError("Point cloud XYZ fields must be FLOAT32")
            offsets[field.name] = field.offset
    if set(offsets) != {"x", "y", "z"}:
        raise ValueError("Point cloud is missing x, y, or z fields")
    data = bytes(cloud.data)
    endian = ">" if cloud.is_bigendian else "<"
    unpack = struct.Struct(endian + "f").unpack_from
    count = int(cloud.width) * int(cloud.height)
    step = max(1, int(stride))
    point_step = int(cloud.point_step)
    for index in range(0, count, step):
        base = index * point_step
        x = unpack(data, base + offsets["x"])[0]
        y = unpack(data, base + offsets["y"])[0]
        z = unpack(data, base + offsets["z"])[0]
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            yield (x, y, z)
