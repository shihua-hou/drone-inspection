"""Online depth-camera occupancy map and comparison with the known park map."""

import json
import math
import os
import struct
import threading
from pathlib import Path
from typing import List, Optional

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Path as PathMessage
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from park_inspection import campus
from park_inspection.exploration import FrontierInformationStructure, plan_exploration
from park_inspection.occupancy import OccupancyMap3D, compare_inspection_plans
from park_inspection.planning import (
    GridPlanner3D,
    Point3,
    flatten_cell_index,
    grid_shape_for_bounds,
    parse_box_obstacles,
    parse_waypoints,
)
from park_inspection.sensing import (
    X500_DEPTH_SENSOR_POSITION_BODY,
    X500_LIDAR_POSITION_BODY,
    cloud_to_map_rays,
    iter_cloud_xyz,
)


class OnlineMapper(Node):
    def __init__(self) -> None:
        super().__init__("online_mapper")
        self.declare_parameter("inspection_points", campus.flat_inspection_points())
        self.declare_parameter("obstacles", campus.flat_obstacles())
        self.declare_parameter("bounds_min", list(campus.BOUNDS_MIN))
        self.declare_parameter("bounds_max", list(campus.BOUNDS_MAX))
        self.declare_parameter("grid_resolution", 1.0)
        self.declare_parameter("obstacle_clearance", 2.5)
        self.declare_parameter("return_altitude", campus.RETURN_ALTITUDE)
        self.declare_parameter("cloud_topic", "/depth_camera/points")
        self.declare_parameter("cloud_convention", "x_forward")
        self.declare_parameter(
            "sensor_position_body", list(X500_DEPTH_SENSOR_POSITION_BODY)
        )
        self.declare_parameter("max_range", 15.0)
        self.declare_parameter("min_range", 0.4)
        self.declare_parameter("ground_z", 0.4)
        self.declare_parameter("cloud_stride", 16)
        self.declare_parameter("max_points_per_cloud", 2500)
        self.declare_parameter("min_cloud_period", 0.5)
        self.declare_parameter("enable_lidar", False)
        self.declare_parameter("lidar_topic", "/lidar/points")
        self.declare_parameter(
            "lidar_position_body", list(X500_LIDAR_POSITION_BODY)
        )
        self.declare_parameter("lidar_stride", 8)
        self.declare_parameter("lidar_max_range", 40.0)
        self.declare_parameter("comparison_period", 5.0)
        self.declare_parameter("map_publish_period", 0.5)
        self.declare_parameter("diagnostics_period", 1.0)
        self.declare_parameter("sensor_timeout", 5.0)
        self.declare_parameter(
            "comparison_file", "~/park_inspection_map_comparison.json"
        )
        self.declare_parameter("enable_exploration", False)

        bounds_min = self._point_parameter("bounds_min")
        bounds_max = self._point_parameter("bounds_max")
        self.inspection_points = parse_waypoints(
            self.get_parameter("inspection_points").value
        )
        self.obstacles = parse_box_obstacles(self.get_parameter("obstacles").value)
        self.resolution = float(self.get_parameter("grid_resolution").value)
        self.clearance = float(self.get_parameter("obstacle_clearance").value)
        self.return_altitude = float(self.get_parameter("return_altitude").value)
        self.convention = str(self.get_parameter("cloud_convention").value)
        if self.convention not in ("x_forward", "optical", "map"):
            raise ValueError(
                "cloud_convention must be 'x_forward', 'optical', or 'map'"
            )
        self.sensor_position_body = self._point_parameter("sensor_position_body")
        self.max_range = float(self.get_parameter("max_range").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.ground_z = float(self.get_parameter("ground_z").value)
        self.cloud_stride = int(self.get_parameter("cloud_stride").value)
        self.max_points_per_cloud = int(self.get_parameter("max_points_per_cloud").value)
        self.min_cloud_period = float(self.get_parameter("min_cloud_period").value)
        self.enable_lidar = bool(self.get_parameter("enable_lidar").value)
        self.enable_exploration = bool(self.get_parameter("enable_exploration").value)
        self.lidar_position_body = self._point_parameter("lidar_position_body")
        self.lidar_stride = int(self.get_parameter("lidar_stride").value)
        self.lidar_max_range = float(self.get_parameter("lidar_max_range").value)
        self.comparison_file = Path(
            os.path.expanduser(str(self.get_parameter("comparison_file").value))
        )
        self.occupancy = OccupancyMap3D(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resolution=self.resolution,
            clearance=self.clearance,
        )
        self.pose: Optional[Point3] = None
        self.orientation = (0.0, 0.0, 0.0, 1.0)
        self.home: Optional[Point3] = None
        self._last_cloud_at: Optional[float] = None
        self._last_lidar_at: Optional[float] = None
        self._map_changed = False
        self._frontier_updates = set()
        self.frontiers = None
        self.exploration_planner = None
        if self.enable_exploration:
            self.frontiers = FrontierInformationStructure()
            self.exploration_planner = GridPlanner3D(
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                obstacles=self.obstacles,
                resolution=self.resolution,
                clearance=self.clearance,
                is_blocked=self.occupancy.blocks,
            )
        self._last_comparison: Optional[dict] = None
        self._comparison_lock = threading.Lock()
        self._comparison_result: Optional[dict] = None
        self._comparison_thread: Optional[threading.Thread] = None
        self._comparison_file_written = False
        self._encoded_occupancy_generation = -1
        self._occupancy_snapshot_data: Optional[str] = None
        self.sensor_timeout = float(self.get_parameter("sensor_timeout").value)
        if self.sensor_timeout <= 0.0:
            raise ValueError("sensor_timeout must be positive")
        self.sensor_stats = {
            "depth": {
                "topic": str(self.get_parameter("cloud_topic").value),
                "frames_received": 0,
                "frames_integrated": 0,
                "accepted_returns": 0,
                "last_received_at": None,
                "last_integrated_at": None,
            },
        }
        if self.enable_lidar:
            self.sensor_stats["lidar"] = {
                "topic": str(self.get_parameter("lidar_topic").value),
                "frames_received": 0,
                "frames_integrated": 0,
                "accepted_returns": 0,
                "last_received_at": None,
                "last_integrated_at": None,
            }

        cloud_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self._pose_callback,
            cloud_qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("cloud_topic").value),
            self._cloud_callback,
            cloud_qos,
        )
        if self.enable_lidar:
            self.create_subscription(
                PointCloud2,
                str(self.get_parameter("lidar_topic").value),
                self._lidar_callback,
                cloud_qos,
            )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/inspection/online_map", latched_qos
        )
        self.lidar_cloud_publisher = None
        if self.enable_lidar:
            self.lidar_cloud_publisher = self.create_publisher(
                PointCloud2, "/inspection/lidar_points", 1
            )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.static_tf_broadcaster.sendTransform(
            [
                self._sensor_transform("lidar_link", self.lidar_position_body),
                self._sensor_transform("depth_sensor", self.sensor_position_body),
            ]
        )
        self.comparison_publisher = self.create_publisher(
            String, "/inspection/map_comparison", latched_qos
        )
        self.occupancy_publisher = self.create_publisher(
            String, "/inspection/online_occupancy", latched_qos
        )
        self.status_publisher = self.create_publisher(
            String, "/inspection/mapper_status", latched_qos
        )
        self.exploration_marker_publisher = None
        self.exploration_path_publisher = None
        if self.enable_exploration:
            self.exploration_marker_publisher = self.create_publisher(
                MarkerArray, "/inspection/exploration_frontiers", latched_qos
            )
            self.exploration_path_publisher = self.create_publisher(
                PathMessage, "/inspection/exploration_path", latched_qos
            )
        period = float(self.get_parameter("comparison_period").value)
        self.create_timer(max(period, 0.5), self._publish_comparison)
        map_period = float(self.get_parameter("map_publish_period").value)
        self.create_timer(max(map_period, 0.2), self._publish_occupancy)
        diagnostics_period = float(self.get_parameter("diagnostics_period").value)
        self.create_timer(max(diagnostics_period, 0.5), self._publish_status)
        if self.enable_lidar:
            self.get_logger().info(
                "Online mapper ready on %s (%s) and %s"
                % (
                    self.get_parameter("cloud_topic").value,
                    self.convention,
                    self.get_parameter("lidar_topic").value,
                )
            )
        else:
            self.get_logger().info(
                "Online mapper ready on %s (%s); lidar disabled"
                % (self.get_parameter("cloud_topic").value, self.convention)
            )

    def _point_parameter(self, name: str) -> Point3:
        values = self.get_parameter(name).value
        if len(values) != 3:
            raise ValueError("%s must contain exactly three coordinates" % name)
        return tuple(float(value) for value in values)

    def _pose_callback(self, message: PoseStamped) -> None:
        position = message.pose.position
        current = (float(position.x), float(position.y), float(position.z))
        orientation = message.pose.orientation
        self.pose = current
        self.orientation = (
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        if self.home is None:
            self.home = current
            self.get_logger().info(
                "Mapper home established at ENU (%.2f, %.2f, %.2f)" % current
            )
        self._publish_base_transform(message)

    def _sensor_transform(self, child: str, translation: Point3) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = child
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.w = 1.0
        return transform

    def _publish_base_transform(self, message: PoseStamped) -> None:
        transform = TransformStamped()
        stamp = message.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
        transform.header.stamp = stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = self.pose[0]
        transform.transform.translation.y = self.pose[1]
        transform.transform.translation.z = self.pose[2]
        transform.transform.rotation = message.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def _publish_map_cloud(self, publisher, points: List[Point3]) -> None:
        cloud = PointCloud2()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = "map"
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * cloud.width
        cloud.is_dense = True
        cloud.data = b"".join(struct.pack("<fff", *point) for point in points)
        publisher.publish(cloud)

    def _cloud_callback(self, message: PointCloud2) -> None:
        now = self._clock_seconds()
        self._record_sensor_message("depth", now)
        if self.pose is None:
            return
        if (
            self._last_cloud_at is not None
            and now - self._last_cloud_at < self.min_cloud_period
        ):
            return
        self._last_cloud_at = now
        try:
            points = list(iter_cloud_xyz(message, self.cloud_stride))
        except ValueError as error:
            self.get_logger().error("Ignoring depth cloud: %s" % error)
            return
        if len(points) > self.max_points_per_cloud:
            step = max(1, len(points) // self.max_points_per_cloud)
            points = points[::step]
        self._integrate_sensor(
            points,
            self.sensor_position_body,
            self.convention,
            self.max_range,
            "depth",
        )

    def _lidar_callback(self, message: PointCloud2) -> None:
        now = self._clock_seconds()
        self._record_sensor_message("lidar", now)
        if self.pose is None:
            return
        if (
            self._last_lidar_at is not None
            and now - self._last_lidar_at < self.min_cloud_period
        ):
            return
        self._last_lidar_at = now
        try:
            points = list(iter_cloud_xyz(message, self.lidar_stride))
        except ValueError as error:
            self.get_logger().error("Ignoring lidar cloud: %s" % error)
            return
        self._integrate_sensor(
            points,
            self.lidar_position_body,
            "x_forward",
            self.lidar_max_range,
            "lidar",
        )

    def _integrate_sensor(
        self,
        points: List[Point3],
        sensor_position_body: Point3,
        convention: str,
        max_range: float,
        sensor_name: str,
    ) -> None:
        origin, endpoints = cloud_to_map_rays(
            points,
            self.pose,
            self.orientation,
            sensor_position_body,
            convention,
        )
        accepted = self.occupancy.integrate_scan(
            origin,
            endpoints,
            max_range=max_range,
            min_range=self.min_range,
            ground_z=self.ground_z,
        )
        if sensor_name == "lidar" and self.lidar_cloud_publisher is not None and endpoints:
            self._publish_map_cloud(self.lidar_cloud_publisher, endpoints)
        stats = self.sensor_stats[sensor_name]
        stats["frames_integrated"] += 1
        stats["accepted_returns"] += accepted
        stats["last_integrated_at"] = self._clock_seconds()
        if accepted:
            self._map_changed = True
            if self.enable_exploration:
                self._frontier_updates.update(self.occupancy.last_updated_cells)

    def _clock_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _record_sensor_message(self, sensor_name: str, now: float) -> None:
        stats = self.sensor_stats[sensor_name]
        stats["frames_received"] += 1
        stats["last_received_at"] = now

    def _publish_status(self) -> None:
        now = self._clock_seconds()
        sensors = {}
        missing_sensors = []
        stale_sensors = []
        for name, stats in self.sensor_stats.items():
            received_at = stats["last_received_at"]
            integrated_at = stats["last_integrated_at"]
            received_age = None
            if received_at is not None:
                received_age = round(max(0.0, now - received_at), 2)
            integrated_age = None
            if integrated_at is not None:
                integrated_age = round(max(0.0, now - integrated_at), 2)
            if received_at is None:
                missing_sensors.append(name)
            elif received_age is not None and received_age > self.sensor_timeout:
                stale_sensors.append(name)
            sensors[name] = {
                "topic": stats["topic"],
                "frames_received": stats["frames_received"],
                "frames_integrated": stats["frames_integrated"],
                "accepted_returns": stats["accepted_returns"],
                "seconds_since_last_frame": received_age,
                "seconds_since_last_integration": integrated_age,
            }

        if self.pose is None:
            status = "waiting_for_pose"
        elif not any(stats["frames_received"] for stats in self.sensor_stats.values()):
            status = "waiting_for_sensor_data"
        elif stale_sensors:
            status = "sensor_data_stale"
        elif missing_sensors:
            status = "partial_sensor_data"
        elif any(
            stats["frames_integrated"] == 0 or stats["accepted_returns"] == 0
            for stats in self.sensor_stats.values()
        ):
            status = "waiting_for_sensor_integration"
        else:
            status = "mapping_active"

        message = String()
        message.data = json.dumps(
            {
                "status": status,
                "pose_received": self.pose is not None,
                "map_observed_cells": len(self.occupancy._log_odds),
                "map_occupied_cells": len(self.occupancy.occupied_cells()),
                "frontier_clusters": (
                    0 if self.frontiers is None else len(self.frontiers.clusters)
                ),
                "sensors": sensors,
            },
            ensure_ascii=False,
        )
        self.status_publisher.publish(message)

    def _route(self) -> List[Point3]:
        home = self.home if self.home is not None else (0.0, 0.0, 0.0)
        return_point = (
            home[0],
            home[1],
            max(home[2], self.return_altitude),
        )
        return [return_point, *self.inspection_points, return_point]

    def _publish_occupancy(self) -> None:
        if self.pose is None:
            return
        generation = self.occupancy.blocked_generation
        if (
            self._occupancy_snapshot_data is None
            or self._encoded_occupancy_generation != generation
        ):
            shape = grid_shape_for_bounds(
                self.occupancy.bounds_min,
                self.occupancy.bounds_max,
                self.resolution,
            )
            blocked_indices = sorted(
                flatten_cell_index(cell, shape)
                for cell in self.occupancy.blocked_cells()
            )
            self._occupancy_snapshot_data = json.dumps(
                {
                    "bounds_min": self.occupancy.bounds_min,
                    "bounds_max": self.occupancy.bounds_max,
                    "resolution": self.resolution,
                    "clearance": self.clearance,
                    "generation": generation,
                    "blocked_indices": blocked_indices,
                },
                separators=(",", ":"),
            )
            self._encoded_occupancy_generation = generation
        message = String()
        message.data = self._occupancy_snapshot_data
        self.occupancy_publisher.publish(message)

    def _publish_comparison(self) -> None:
        if self.pose is None:
            return
        with self._comparison_lock:
            comparison = self._comparison_result
            self._comparison_result = None
        if comparison is not None:
            self._last_comparison = comparison
            self._comparison_file_written = False
            message = String()
            message.data = json.dumps(comparison, ensure_ascii=False)
            self.comparison_publisher.publish(message)
        if self._map_changed or self._last_comparison is None:
            self._publish_occupied_markers()
            if self._comparison_thread is None or not self._comparison_thread.is_alive():
                self._map_changed = False
                self._comparison_thread = threading.Thread(
                    target=self._run_comparison,
                    args=(self.occupancy.snapshot(), self._route()),
                    daemon=True,
                )
                self._comparison_thread.start()

        if self._last_comparison is not None and not self._comparison_file_written:
            try:
                self.comparison_file.parent.mkdir(parents=True, exist_ok=True)
                self.comparison_file.write_text(
                    json.dumps(self._last_comparison, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as error:
                self.get_logger().error(
                    "Failed to write map comparison to %s: %s"
                    % (self.comparison_file, error)
                )
            else:
                self._comparison_file_written = True
                self.get_logger().info(
                    "Map comparison saved to %s" % self.comparison_file
                )
        self._publish_exploration()

    def _run_comparison(self, occupancy: OccupancyMap3D, route: List[Point3]) -> None:
        try:
            comparison = compare_inspection_plans(
                occupancy.bounds_min,
                occupancy.bounds_max,
                self.obstacles,
                self.resolution,
                self.clearance,
                occupancy,
                route,
            )
        except Exception as error:  # noqa: BLE001 - report and keep mapping
            self.get_logger().error("Map comparison failed: %r" % (error,))
            return
        with self._comparison_lock:
            self._comparison_result = comparison

    def _publish_occupied_markers(self) -> None:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "online_occupancy"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = self.resolution * 0.9
        marker.color.r = 0.1
        marker.color.g = 0.75
        marker.color.b = 0.85
        marker.color.a = 0.85
        cells = self.occupancy.occupied_cells()
        if not cells:
            marker.action = Marker.DELETE
        else:
            marker.action = Marker.ADD
            for cell in cells:
                center = self.occupancy.cell_center(cell)
                marker.points.append(Point(x=center[0], y=center[1], z=center[2]))
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_publisher.publish(array)

    def _publish_exploration(self) -> None:
        if not self.enable_exploration or self.frontiers is None:
            return
        if self._frontier_updates:
            self.frontiers.update(self.occupancy, tuple(self._frontier_updates))
            self._frontier_updates.clear()
        plan = None
        if self.pose is not None and self.frontiers.clusters:
            plan = plan_exploration(
                self.pose,
                self._yaw(),
                self.frontiers.clusters,
                self.exploration_planner,
            )
        self._publish_frontier_markers()
        path = PathMessage()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        if plan is not None:
            for point in plan.path:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.position.z = point[2]
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
        self.exploration_path_publisher.publish(path)

    def _yaw(self) -> float:
        x_value, y_value, z_value, w_value = self.orientation
        return math.atan2(
            2.0 * (w_value * z_value + x_value * y_value),
            1.0 - 2.0 * (y_value * y_value + z_value * z_value),
        )

    def _publish_frontier_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        cells = Marker()
        cells.header.frame_id = "map"
        cells.header.stamp = stamp
        cells.ns = "frontiers"
        cells.id = 0
        cells.type = Marker.CUBE_LIST
        cells.pose.orientation.w = 1.0
        cells.scale.x = cells.scale.y = cells.scale.z = self.resolution * 0.45
        cells.color.r = 0.95
        cells.color.g = 0.55
        cells.color.b = 0.1
        cells.color.a = 0.9
        viewpoints = Marker()
        viewpoints.header.frame_id = "map"
        viewpoints.header.stamp = stamp
        viewpoints.ns = "viewpoints"
        viewpoints.id = 1
        viewpoints.type = Marker.SPHERE_LIST
        viewpoints.pose.orientation.w = 1.0
        viewpoints.scale.x = viewpoints.scale.y = viewpoints.scale.z = 0.6
        viewpoints.color.r = 0.95
        viewpoints.color.g = 0.85
        viewpoints.color.b = 0.2
        viewpoints.color.a = 0.95
        frontier_cells = [
            cell for cluster in self.frontiers.clusters for cell in cluster.cells
        ]
        if len(frontier_cells) > 2000:
            frontier_cells = frontier_cells[:: max(1, len(frontier_cells) // 2000)]
        if frontier_cells:
            cells.action = Marker.ADD
            for cell in frontier_cells:
                center = self.occupancy.cell_center(cell)
                cells.points.append(Point(x=center[0], y=center[1], z=center[2]))
        else:
            cells.action = Marker.DELETE
        if self.frontiers.clusters:
            viewpoints.action = Marker.ADD
            for cluster in self.frontiers.clusters:
                position = cluster.viewpoints[0].position
                viewpoints.points.append(
                    Point(x=position[0], y=position[1], z=position[2])
                )
        else:
            viewpoints.action = Marker.DELETE
        array = MarkerArray()
        array.markers.extend((cells, viewpoints))
        self.exploration_marker_publisher.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OnlineMapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
