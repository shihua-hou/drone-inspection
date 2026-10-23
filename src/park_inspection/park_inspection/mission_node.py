"""MAVROS Offboard mission runner for a known-map park inspection."""

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Path as PathMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from park_inspection import campus
from park_inspection.planning import (
    BoxObstacle,
    GridPlanner3D,
    Point3,
    grid_shape_for_bounds,
    parse_box_obstacles,
    parse_waypoints,
    path_intersects_grid_cells,
    point_in_grid_cells,
    unflatten_cell_index,
)


class InspectionMission(Node):
    def __init__(self) -> None:
        super().__init__("inspection_mission")
        self.declare_parameter("inspection_points", campus.flat_inspection_points())
        self.declare_parameter("obstacles", campus.flat_obstacles())
        self.declare_parameter("bounds_min", list(campus.BOUNDS_MIN))
        self.declare_parameter("bounds_max", list(campus.BOUNDS_MAX))
        self.declare_parameter("grid_resolution", 1.0)
        self.declare_parameter("obstacle_clearance", 2.5)
        self.declare_parameter("minimum_safe_clearance", 1.5)
        self.declare_parameter("acceptance_radius", 1.0)
        self.declare_parameter("return_altitude", campus.RETURN_ALTITUDE)
        self.declare_parameter("metrics_file", "~/park_inspection_metrics.json")
        self.declare_parameter("enable_online_replanning", False)
        self.declare_parameter("online_map_timeout", 2.5)
        self.declare_parameter("online_replan_timeout", 10.0)

        points_flat = self.get_parameter("inspection_points").value
        obstacles_flat = self.get_parameter("obstacles").value
        bounds_min = self._point_parameter("bounds_min")
        bounds_max = self._point_parameter("bounds_max")
        self.inspection_points: List[Point3] = parse_waypoints(points_flat)
        obstacles = parse_box_obstacles(obstacles_flat)
        self.planner = GridPlanner3D(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            obstacles=obstacles,
            resolution=float(self.get_parameter("grid_resolution").value),
            clearance=float(self.get_parameter("obstacle_clearance").value),
        )
        self.acceptance_radius = float(self.get_parameter("acceptance_radius").value)
        self.return_altitude = float(self.get_parameter("return_altitude").value)
        self.minimum_safe_clearance = float(
            self.get_parameter("minimum_safe_clearance").value
        )
        self.enable_online_replanning = bool(
            self.get_parameter("enable_online_replanning").value
        )
        self.online_map_timeout = float(
            self.get_parameter("online_map_timeout").value
        )
        self.online_replan_timeout = float(
            self.get_parameter("online_replan_timeout").value
        )
        if self.minimum_safe_clearance < 0.0:
            raise ValueError("minimum_safe_clearance cannot be negative")
        if not math.isfinite(self.online_map_timeout) or self.online_map_timeout <= 0.0:
            raise ValueError("online_map_timeout must be positive and finite")
        if (
            not math.isfinite(self.online_replan_timeout)
            or self.online_replan_timeout <= 0.0
        ):
            raise ValueError("online_replan_timeout must be positive and finite")
        self.metrics_file = Path(
            os.path.expanduser(str(self.get_parameter("metrics_file").value))
        )

        self.pose: Optional[Point3] = None
        self.home: Optional[Point3] = None
        self.previous_pose: Optional[Point3] = None
        self.state = State()
        self.route: List[Point3] = []
        self.route_paths: List[List[Point3]] = []
        self.route_prepared = False
        self.route_index = 0
        self.path: Optional[List[Point3]] = None
        self.path_index = 0
        self.prestream_count = 0
        self.mission_started_at: Optional[float] = None
        self.mission_finished = False
        self.mission_failed = False
        self.distance_m = 0.0
        self.minimum_clearance_m = float("inf")
        self.minimum_clearance_position: Optional[Point3] = None
        self.safety_margin_violations = 0
        self._inside_safety_margin = False
        self._mode_request_inflight = False
        self._arm_request_inflight = False
        self._next_mode_request_at = 0.0
        self._next_arm_request_at = 0.0
        self.online_mapper_status = "waiting_for_mapper"
        self.online_mapper_status_at: Optional[float] = None
        self.online_blocked_cells: Set[Tuple[int, int, int]] = set()
        self.online_map_received_at: Optional[float] = None
        self.online_map_generation = 0
        self.online_replan_in_progress = False
        self.online_replan_started_at: Optional[float] = None
        self.online_replan_result: Optional[
            Tuple[int, int, Optional[List[Point3]], Optional[str], float]
        ] = None
        self.online_replan_lock = threading.Lock()
        self.online_replans = 0
        self.online_replan_failures = 0
        self.online_replan_attempts = 0
        self.online_replan_timeouts = 0
        self.online_replan_duration_total_s = 0.0
        self.online_replan_duration_max_s = 0.0
        self.online_replan_duration_last_s: Optional[float] = None
        self.online_map_timeout_events = 0
        self.failure_reason: Optional[str] = None
        self.failure_mode_at_detection: Optional[str] = None
        self.failure_mode_after_detection: Optional[str] = None
        self._last_setpoint_at: Optional[float] = None
        self._max_setpoint_gap_s = 0.0
        self._setpoint_gaps_over_100ms = 0
        self._setpoint_publish_count = 0
        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        pose_qos = QoSProfile(
            depth=5, reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self._pose_callback,
            pose_qos,
        )
        self.state_subscriber = self.create_subscription(
            State, "/mavros/state", self._state_callback, 10
        )
        if self.enable_online_replanning:
            self.online_map_subscriber = self.create_subscription(
                String,
                "/inspection/online_occupancy",
                self._online_map_callback,
                latched_qos,
            )
            self.mapper_status_subscriber = self.create_subscription(
                String,
                "/inspection/mapper_status",
                self._mapper_status_callback,
                latched_qos,
            )
        self.setpoint_publisher = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", 10
        )
        self.path_publisher = self.create_publisher(
            PathMessage, "/inspection/planned_path", latched_qos
        )
        self.markers_publisher = self.create_publisher(
            MarkerArray, "/inspection/map_markers", latched_qos
        )
        self.metrics_publisher = self.create_publisher(
            String, "/inspection/metrics", latched_qos
        )
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.timer = self.create_timer(0.05, self._tick)
        self.marker_timer = self.create_timer(
            1.0, lambda: self._publish_map_markers(obstacles)
        )
        self._publish_map_markers(obstacles)
        self.get_logger().info(
            "Inspection mission ready: %d targets, %.1f m grid, %.1f m clearance"
            % (
                len(self.inspection_points),
                self.planner.resolution,
                self.planner.clearance,
            )
        )

    def _point_parameter(self, name: str) -> Point3:
        values = self.get_parameter(name).value
        if len(values) != 3:
            raise ValueError("%s must contain exactly three coordinates" % name)
        return tuple(float(value) for value in values)

    def _pose_callback(self, message: PoseStamped) -> None:
        position = message.pose.position
        current = (float(position.x), float(position.y), float(position.z))
        if self.home is None:
            self.home = current
            self.get_logger().info(
                "Home established at ENU (%.2f, %.2f, %.2f)" % current
            )
        if self.mission_started_at is not None and self.previous_pose is not None:
            self.distance_m += math.dist(self.previous_pose, current)
        self.previous_pose = current
        self.pose = current

        clearance = self.planner.signed_clearance(current)
        if clearance < self.minimum_clearance_m:
            self.minimum_clearance_m = clearance
            self.minimum_clearance_position = current
        within_margin = clearance < self.minimum_safe_clearance
        if within_margin and not self._inside_safety_margin:
            self.safety_margin_violations += 1
        self._inside_safety_margin = within_margin

    def _state_callback(self, message: State) -> None:
        previous_mode = self.state.mode
        self.state = message
        if (
            self.mission_failed
            and self.failure_mode_at_detection is not None
            and message.mode != previous_mode
            and message.mode != self.failure_mode_at_detection
        ):
            self.failure_mode_after_detection = message.mode
            self._write_metrics("failed")

    def _online_map_callback(self, message: String) -> None:
        try:
            snapshot = json.loads(message.data)
            bounds_min = tuple(float(value) for value in snapshot["bounds_min"])
            bounds_max = tuple(float(value) for value in snapshot["bounds_max"])
            resolution = float(snapshot["resolution"])
            clearance = float(snapshot["clearance"])
            raw_indices = snapshot["blocked_indices"]
            if len(bounds_min) != 3 or len(bounds_max) != 3:
                raise ValueError("online map bounds must contain three coordinates")
            if (
                any(not math.isfinite(value) for value in bounds_min + bounds_max)
                or not math.isclose(resolution, self.planner.resolution)
                or not math.isclose(clearance, self.planner.clearance)
                or any(
                    not math.isclose(bounds_min[axis], self.planner.bounds_min[axis])
                    or not math.isclose(bounds_max[axis], self.planner.bounds_max[axis])
                    for axis in range(3)
                )
            ):
                raise ValueError("online map geometry does not match the planner")
            if not isinstance(raw_indices, list):
                raise ValueError("blocked_indices must be a list")
            shape = grid_shape_for_bounds(
                self.planner.bounds_min,
                self.planner.bounds_max,
                self.planner.resolution,
            )
            maximum_cells = math.prod(shape)
            if len(raw_indices) > maximum_cells:
                raise ValueError("online map contains more cells than its grid")
            blocked_cells = set()
            for raw_index in raw_indices:
                blocked_cells.add(unflatten_cell_index(raw_index, shape))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error("Ignoring invalid online occupancy map: %s" % error)
            return

        if blocked_cells != self.online_blocked_cells:
            self.online_blocked_cells = blocked_cells
            self.online_map_generation += 1
        self.online_map_received_at = time.monotonic()

    def _mapper_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)["status"]
            if not isinstance(status, str):
                raise ValueError("mapper status must be a string")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error("Ignoring invalid mapper status: %s" % error)
            self.online_mapper_status = "invalid_mapper_status"
            self.online_mapper_status_at = time.monotonic()
            return
        self.online_mapper_status = status
        self.online_mapper_status_at = time.monotonic()

    def _online_map_is_ready(self) -> bool:
        now = time.monotonic()
        return (
            self.online_mapper_status == "mapping_active"
            and self.online_mapper_status_at is not None
            and now - self.online_mapper_status_at <= self.online_map_timeout
            and self.online_map_received_at is not None
            and now - self.online_map_received_at <= self.online_map_timeout
        )

    def _tick(self) -> None:
        if self.pose is None:
            return
        if self.mission_finished:
            if self.state.mode == "OFFBOARD":
                self._publish_setpoint(self.pose)
                self._request_mode("AUTO.LAND")
            return
        if self.mission_failed:
            if self.state.mode == "OFFBOARD":
                self._publish_setpoint(self.pose)
                self._request_mode("AUTO.LOITER")
            return

        if self.home is None:
            return

        if not self.route_prepared:
            if not self.state.connected:
                return
            self._prepare_route()
            return

        target = self.pose
        if self.state.connected and self.state.mode == "OFFBOARD" and self.state.armed:
            if self.mission_started_at is None:
                self.mission_started_at = time.monotonic()
                self.previous_pose = self.pose
                self.get_logger().info("Offboard mission started")
            if self.enable_online_replanning and not self._online_map_is_ready():
                self.online_map_timeout_events += 1
                self._fail_mission(
                    "Online mapping became unavailable; holding position and requesting LOITER"
                )
            if self.mission_failed:
                target = self.pose
            elif self.enable_online_replanning:
                self._process_online_replan_result()
                if self.mission_failed or self.online_replan_in_progress:
                    target = self.pose
                else:
                    target = self._mission_target()
                    if self._path_blocked_by_online_map():
                        self._start_online_replan()
                        target = self.pose
            else:
                target = self._mission_target()
        elif self.state.connected:
            self.prestream_count += 1
            if self.prestream_count >= 40 and self.state.mode != "OFFBOARD":
                self._request_mode("OFFBOARD")
            elif self.state.mode == "OFFBOARD" and not self.state.armed:
                if not self.enable_online_replanning or self._online_map_is_ready():
                    self._request_arm()

        self._publish_setpoint(target)

    def _publish_setpoint(self, target: Point3) -> None:
        now = time.monotonic()
        if self._last_setpoint_at is not None:
            gap = now - self._last_setpoint_at
            self._max_setpoint_gap_s = max(self._max_setpoint_gap_s, gap)
            if gap > 0.1:
                self._setpoint_gaps_over_100ms += 1
        self._last_setpoint_at = now
        self._setpoint_publish_count += 1
        setpoint = PoseStamped()
        setpoint.header.stamp = self.get_clock().now().to_msg()
        setpoint.header.frame_id = "map"
        setpoint.pose.position.x = target[0]
        setpoint.pose.position.y = target[1]
        setpoint.pose.position.z = target[2]
        setpoint.pose.orientation.w = 1.0
        self.setpoint_publisher.publish(setpoint)

    def _path_blocked_by_online_map(self) -> bool:
        if self.path is None or self.path_index >= len(self.path):
            return False
        return path_intersects_grid_cells(
            self.pose,
            self.path[self.path_index:],
            self.planner.bounds_min,
            self.planner.bounds_max,
            self.planner.resolution,
            self.online_blocked_cells,
        )

    def _start_online_replan(self) -> None:
        if self.online_replan_in_progress or self.route_index >= len(self.route):
            return
        start = self.pose
        goal = self.route[self.route_index]
        blocked_cells = set(self.online_blocked_cells)
        generation = self.online_map_generation
        route_index = self.route_index
        obstacles = self.planner.obstacles
        bounds_min = self.planner.bounds_min
        bounds_max = self.planner.bounds_max
        resolution = self.planner.resolution
        clearance = self.planner.clearance
        self.online_replan_in_progress = True
        self.online_replan_started_at = time.monotonic()
        self.online_replan_attempts += 1
        self.get_logger().warning(
            "Online obstacle intersects the active route; holding and replanning "
            "to target %d/%d" % (route_index + 1, len(self.route))
        )
        worker = threading.Thread(
            target=self._compute_online_replan,
            args=(
                generation,
                route_index,
                start,
                goal,
                blocked_cells,
                obstacles,
                bounds_min,
                bounds_max,
                resolution,
                clearance,
            ),
            daemon=True,
        )
        worker.start()

    def _compute_online_replan(
        self,
        generation: int,
        route_index: int,
        start: Point3,
        goal: Point3,
        blocked_cells: Set[Tuple[int, int, int]],
        obstacles: Sequence[BoxObstacle],
        bounds_min: Point3,
        bounds_max: Point3,
        resolution: float,
        clearance: float,
    ) -> None:
        started_at = time.monotonic()
        try:
            planner = GridPlanner3D(
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                obstacles=obstacles,
                resolution=resolution,
                clearance=clearance,
                is_blocked=lambda point: point_in_grid_cells(
                    point,
                    bounds_min,
                    bounds_max,
                    resolution,
                    blocked_cells,
                ),
            )
            path = planner.plan(start, goal)
            error = None
        except Exception as exception:
            path = None
            error = "%s: %s" % (type(exception).__name__, exception)
        duration = time.monotonic() - started_at
        with self.online_replan_lock:
            self.online_replan_result = (
                generation,
                route_index,
                path,
                error,
                duration,
            )

    def _process_online_replan_result(self) -> None:
        with self.online_replan_lock:
            result = self.online_replan_result
            self.online_replan_result = None
        if result is None:
            if (
                self.online_replan_in_progress
                and self.online_replan_started_at is not None
                and time.monotonic() - self.online_replan_started_at
                >= self.online_replan_timeout
            ):
                duration = time.monotonic() - self.online_replan_started_at
                self.online_replan_in_progress = False
                self.online_replan_started_at = None
                self.online_replan_failures += 1
                self.online_replan_timeouts += 1
                self.online_replan_duration_last_s = duration
                self.online_replan_duration_total_s += duration
                self.online_replan_duration_max_s = max(
                    self.online_replan_duration_max_s, duration
                )
                self._fail_mission(
                    "Online replanning exceeded %.1f seconds; holding and requesting LOITER"
                    % self.online_replan_timeout
                )
            return

        generation, route_index, path, error, duration = result
        self.online_replan_in_progress = False
        self.online_replan_started_at = None
        self.online_replan_duration_last_s = duration
        self.online_replan_duration_total_s += duration
        self.online_replan_duration_max_s = max(
            self.online_replan_duration_max_s, duration
        )
        if route_index != self.route_index:
            return
        if generation != self.online_map_generation:
            self._start_online_replan()
            return
        if error is not None or path is None:
            self.online_replan_failures += 1
            self._fail_mission(
                "No safe route exists on the fused map; holding and requesting LOITER"
                + (": %s" % error if error else "")
            )
            return

        self.path = path
        self.path_index = 0
        self.online_replans += 1
        self._publish_path(path)
        self.get_logger().info(
            "Online replan %d accepted with %d path points"
            % (self.online_replans, len(path))
        )

    def _fail_mission(self, reason: str) -> None:
        if self.mission_failed:
            return
        self.mission_failed = True
        self.failure_reason = reason
        self.failure_mode_at_detection = self.state.mode
        self.get_logger().error(reason)
        self._write_metrics("failed")

    def _mission_target(self) -> Point3:
        if self.route_index >= len(self.route):
            self._finish_mission()
            return self.pose
        if self.path is None:
            self._plan_to_next_target()
        if self.mission_failed or self.path is None:
            return self.pose

        target = self.path[self.path_index]
        if math.dist(self.pose, target) <= self.acceptance_radius:
            self.path_index += 1
            if self.path_index >= len(self.path):
                completed_target = self.route[self.route_index]
                self.route_index += 1
                self.path = None
                self.path_index = 0
                if self.route_index == 1:
                    label = "Takeoff altitude reached"
                elif self.route_index <= len(self.inspection_points) + 1:
                    label = "Reached inspection target %d/%d" % (
                        self.route_index - 1,
                        len(self.inspection_points),
                    )
                else:
                    label = "Returned to home"
                self.get_logger().info(
                    "%s at (%.1f, %.1f, %.1f)"
                    % (label, completed_target[0], completed_target[1], completed_target[2])
                )
                if self.route_index >= len(self.route):
                    self._finish_mission()
                    return self.pose
                self._plan_to_next_target()
            target = self.path[self.path_index]
        return target

    def _plan_to_next_target(self) -> None:
        self.path = self.route_paths[self.route_index]
        self.path_index = 0
        self._publish_path(self.path)
        self.get_logger().info(
            "Loaded preplanned target %d/%d with %d path points"
            % (self.route_index + 1, len(self.route), len(self.path))
        )

    def _publish_path(self, path: List[Point3]) -> None:
        path_message = PathMessage()
        path_message.header.stamp = self.get_clock().now().to_msg()
        path_message.header.frame_id = "map"
        for point in path:
            pose = PoseStamped()
            pose.header = path_message.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.w = 1.0
            path_message.poses.append(pose)
        self.path_publisher.publish(path_message)

    def _prepare_route(self) -> None:
        return_point = (
            self.home[0],
            self.home[1],
            max(self.home[2], self.return_altitude),
        )
        self.route = [return_point] + list(self.inspection_points) + [return_point]
        start = self.pose
        try:
            for goal in self.route:
                self.route_paths.append(self.planner.plan(start, goal))
                start = goal
        except (ValueError, RuntimeError) as error:
            self.route_paths.clear()
            self._fail_mission(
                "Mission planning failed before Offboard: %s" % error
            )
            return

        self.route_prepared = True
        self.get_logger().info(
            "Preplanned %d route legs before requesting Offboard"
            % len(self.route_paths)
        )
        self._plan_to_next_target()

    def _request_mode(self, mode: str) -> None:
        now = time.monotonic()
        if (
            self._mode_request_inflight
            or now < self._next_mode_request_at
            or not self.mode_client.service_is_ready()
        ):
            return
        request = SetMode.Request()
        request.base_mode = 0
        request.custom_mode = mode
        self._mode_request_inflight = True
        self._next_mode_request_at = now + 3.0
        future = self.mode_client.call_async(request)
        future.add_done_callback(
            lambda result: self._mode_response(result, mode)
        )

    def _mode_response(self, future, mode: str) -> None:
        self._mode_request_inflight = False
        response = future.result()
        if response.mode_sent:
            self.get_logger().info("Requested PX4 mode %s" % mode)
        else:
            self.get_logger().warning("PX4 did not accept mode request %s" % mode)

    def _request_arm(self) -> None:
        now = time.monotonic()
        if (
            self._arm_request_inflight
            or now < self._next_arm_request_at
            or not self.arm_client.service_is_ready()
        ):
            return
        request = CommandBool.Request()
        request.value = True
        self._arm_request_inflight = True
        self._next_arm_request_at = now + 3.0
        future = self.arm_client.call_async(request)
        future.add_done_callback(self._arm_response)

    def _arm_response(self, future) -> None:
        self._arm_request_inflight = False
        response = future.result()
        if response.success:
            self.get_logger().info("PX4 arm command accepted")
        else:
            self.get_logger().warning(
                "PX4 rejected arming; check the PX4 console for failed preflight checks"
            )

    def _finish_mission(self) -> None:
        if self.mission_finished:
            return
        self.mission_finished = True
        self._write_metrics("completed")
        self.get_logger().info("Inspection complete; requesting AUTO.LAND")
        self._request_mode("AUTO.LAND")

    def _write_metrics(self, status: str) -> None:
        elapsed = (
            time.monotonic() - self.mission_started_at
            if self.mission_started_at is not None
            else 0.0
        )
        metrics = {
            "status": status,
            "targets_completed": min(
                max(self.route_index - 1, 0), len(self.inspection_points)
            ),
            "targets_total": len(self.inspection_points),
            "coverage": min(max(self.route_index - 1, 0), len(self.inspection_points))
            / len(self.inspection_points),
            "flight_distance_m": round(self.distance_m, 2),
            "elapsed_time_s": round(elapsed, 2),
            "minimum_obstacle_clearance_m": (
                None
                if math.isinf(self.minimum_clearance_m)
                else round(self.minimum_clearance_m, 2)
            ),
            "minimum_clearance_position_enu": (
                None
                if self.minimum_clearance_position is None
                else [round(value, 2) for value in self.minimum_clearance_position]
            ),
            "minimum_safe_clearance_m": self.minimum_safe_clearance,
            "safety_margin_violations": self.safety_margin_violations,
            "online_replans": self.online_replans,
            "online_replan_failures": self.online_replan_failures,
            "online_replan_attempts": self.online_replan_attempts,
            "online_replan_timeouts": self.online_replan_timeouts,
            "online_replan_duration_last_s": (
                None
                if self.online_replan_duration_last_s is None
                else round(self.online_replan_duration_last_s, 3)
            ),
            "online_replan_duration_average_s": (
                None
                if self.online_replan_attempts == 0
                else round(
                    self.online_replan_duration_total_s
                    / self.online_replan_attempts,
                    3,
                )
            ),
            "online_replan_duration_max_s": round(
                self.online_replan_duration_max_s, 3
            ),
            "online_map_timeout_events": self.online_map_timeout_events,
            "failure_reason": self.failure_reason,
            "failure_mode_at_detection": self.failure_mode_at_detection,
            "failure_mode_after_detection": self.failure_mode_after_detection,
            "setpoint_publish_count": self._setpoint_publish_count,
            "max_setpoint_gap_s": (
                None
                if self._setpoint_publish_count < 2
                else round(self._max_setpoint_gap_s, 4)
            ),
            "setpoint_gaps_over_100ms": self._setpoint_gaps_over_100ms,
        }
        message = String()
        message.data = json.dumps(metrics, ensure_ascii=False)
        self.metrics_publisher.publish(message)
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_file.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            self.get_logger().error(
                "Failed to write metrics to %s: %s"
                % (self.metrics_file, error)
            )
        self.get_logger().info("Mission metrics: %s" % message.data)

    def _publish_map_markers(self, obstacles: List[BoxObstacle]) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for index, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "park_obstacles"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.center[0]
            marker.pose.position.y = obstacle.center[1]
            marker.pose.position.z = obstacle.center[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = obstacle.size
            marker.color.r = 0.75
            marker.color.g = 0.28
            marker.color.b = 0.16
            marker.color.a = 0.65
            markers.markers.append(marker)

        for index, target in enumerate(self.inspection_points):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "inspection_targets"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = target[0]
            marker.pose.position.y = target[1]
            marker.pose.position.z = target[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            marker.color.r = 0.15
            marker.color.g = 0.9
            marker.color.b = 0.25
            marker.color.a = 0.9
            markers.markers.append(marker)
        self.markers_publisher.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InspectionMission()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
