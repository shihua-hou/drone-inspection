"""Collision-aware grid planning for a 3D park inspection map."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count, product
from math import ceil, dist, floor, prod, sqrt
from typing import AbstractSet, Callable, Dict, List, Optional, Sequence, Tuple

Point3 = Tuple[float, float, float]
Cell = Tuple[int, int, int]
_BUCKET_SIZE = 4.0


def grid_shape_for_bounds(
    bounds_min: Point3, bounds_max: Point3, resolution: float
) -> Cell:
    """Return the number of voxels on each axis for a bounded grid."""
    if resolution <= 0.0 or any(bounds_min[i] >= bounds_max[i] for i in range(3)):
        raise ValueError("Grid geometry must have positive extents and resolution")
    shape = tuple(
        int((bounds_max[axis] - bounds_min[axis]) / resolution)
        for axis in range(3)
    )
    if any(count <= 0 for count in shape):
        raise ValueError("Grid bounds are smaller than one voxel")
    return shape


def flatten_cell_index(cell: Cell, shape: Cell) -> int:
    """Encode an x/y/z voxel coordinate as one row-major integer."""
    if any(type(count) is not int or count <= 0 for count in shape):
        raise ValueError("Grid shape must contain positive integers")
    if any(
        type(cell[axis]) is not int or cell[axis] < 0 or cell[axis] >= shape[axis]
        for axis in range(3)
    ):
        raise ValueError("Voxel coordinate is outside the grid")
    return (cell[0] * shape[1] + cell[1]) * shape[2] + cell[2]


def unflatten_cell_index(index: int, shape: Cell) -> Cell:
    """Decode a row-major voxel index into x/y/z coordinates."""
    if any(type(count) is not int or count <= 0 for count in shape):
        raise ValueError("Grid shape must contain positive integers")
    if type(index) is not int or index < 0 or index >= prod(shape):
        raise ValueError("Voxel index is outside the grid")
    z = index % shape[2]
    y = (index // shape[2]) % shape[1]
    x = index // (shape[1] * shape[2])
    return x, y, z


@dataclass(frozen=True)
class BoxObstacle:
    center: Point3
    size: Point3

    def __post_init__(self) -> None:
        if any(length <= 0.0 for length in self.size):
            raise ValueError("Obstacle dimensions must be positive")

    def signed_distance(self, point: Point3) -> float:
        delta = tuple(
            abs(point[axis] - self.center[axis]) - self.size[axis] / 2.0
            for axis in range(3)
        )
        outside = tuple(max(value, 0.0) for value in delta)
        if any(value > 0.0 for value in delta):
            return sqrt(sum(value * value for value in outside))
        return max(delta)

    def contains(self, point: Point3, margin: float = 0.0) -> bool:
        return all(
            abs(point[axis] - self.center[axis])
            <= self.size[axis] / 2.0 + margin
            for axis in range(3)
        )


class GridPlanner3D:
    """A* planner over a bounded 26-connected grid with line-of-sight smoothing."""

    def __init__(
        self,
        bounds_min: Point3,
        bounds_max: Point3,
        obstacles: Sequence[BoxObstacle],
        resolution: float = 1.0,
        clearance: float = 1.0,
        is_blocked: Optional[Callable[[Point3], bool]] = None,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("Grid resolution must be positive")
        if clearance < 0.0:
            raise ValueError("Obstacle clearance cannot be negative")
        if any(bounds_min[i] >= bounds_max[i] for i in range(3)):
            raise ValueError("Each minimum bound must be less than its maximum")
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.obstacles = tuple(obstacles)
        self.resolution = resolution
        self.clearance = clearance
        self.is_blocked = is_blocked
        self._max_index = tuple(
            int((bounds_max[i] - bounds_min[i]) / resolution)
            for i in range(3)
        )
        self._neighbors = tuple(
            offset
            for offset in product((-1, 0, 1), repeat=3)
            if offset != (0, 0, 0)
        )
        self._buckets: Dict[Tuple[int, int], List[BoxObstacle]] = {}
        for obstacle in self.obstacles:
            low = [
                floor((obstacle.center[axis] - obstacle.size[axis] / 2.0 - clearance) / _BUCKET_SIZE)
                for axis in range(2)
            ]
            high = [
                floor((obstacle.center[axis] + obstacle.size[axis] / 2.0 + clearance) / _BUCKET_SIZE)
                for axis in range(2)
            ]
            for bucket_x in range(low[0], high[0] + 1):
                for bucket_y in range(low[1], high[1] + 1):
                    self._buckets.setdefault((bucket_x, bucket_y), []).append(obstacle)
        self._cell_clear: Dict[Cell, bool] = {}

    def point_is_free(self, point: Point3) -> bool:
        if any(
            point[i] < self.bounds_min[i] or point[i] > self.bounds_max[i]
            for i in range(3)
        ):
            return False
        if not self._clear_of_boxes(point):
            return False
        return self.is_blocked is None or not self.is_blocked(point)

    def _clear_of_boxes(self, point: Point3) -> bool:
        nearby = self._buckets.get(
            (floor(point[0] / _BUCKET_SIZE), floor(point[1] / _BUCKET_SIZE)), ()
        )
        return not any(obstacle.contains(point, self.clearance) for obstacle in nearby)

    def _cell_is_free(self, cell: Cell) -> bool:
        clear = self._cell_clear.get(cell)
        point = self._cell_to_point(cell)
        if clear is None:
            clear = all(
                self.bounds_min[i] <= point[i] <= self.bounds_max[i] for i in range(3)
            ) and self._clear_of_boxes(point)
            self._cell_clear[cell] = clear
        if not clear:
            return False
        return self.is_blocked is None or not self.is_blocked(point)

    def signed_clearance(self, point: Point3) -> float:
        if not self.obstacles:
            return float("inf")
        return min(obstacle.signed_distance(point) for obstacle in self.obstacles)

    def plan(self, start: Point3, goal: Point3) -> List[Point3]:
        if not self.point_is_free(start):
            raise ValueError("Start position is outside bounds or inside an obstacle")
        if not self.point_is_free(goal):
            raise ValueError("Goal position is outside bounds or inside an obstacle")

        start_cell = self._point_to_cell(start)
        goal_cell = self._point_to_cell(goal)
        if start_cell == goal_cell:
            if not self._segment_is_free(start, goal):
                raise RuntimeError("No collision-free route between start and goal")
            return [start] if start == goal else [start, goal]

        frontier = []
        sequence = count()
        heappush(
            frontier,
            (self._heuristic(start_cell, goal_cell), next(sequence), start_cell),
        )
        came_from = {}
        costs = {start_cell: 0.0}
        closed = set()

        while frontier:
            _, _, current = heappop(frontier)
            if current in closed:
                continue
            closed.add(current)
            if current == goal_cell:
                break

            current_point = self._cell_to_point(current)
            for offset in self._neighbors:
                neighbor = tuple(current[i] + offset[i] for i in range(3))
                if any(
                    neighbor[i] < 0 or neighbor[i] > self._max_index[i]
                    for i in range(3)
                ):
                    continue
                if not self._cell_is_free(neighbor):
                    continue
                neighbor_point = self._cell_to_point(neighbor)
                if not self._step_is_free(current_point, neighbor_point):
                    continue

                edge_cost = dist(current_point, neighbor_point)
                new_cost = costs[current] + edge_cost
                if new_cost >= costs.get(neighbor, float("inf")):
                    continue
                costs[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + self._heuristic(neighbor, goal_cell)
                heappush(frontier, (priority, next(sequence), neighbor))
        else:
            raise RuntimeError("No collision-free route found within planning bounds")

        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(came_from[cells[-1]])
        cells.reverse()
        anchors = [start]
        anchors.extend(self._cell_to_point(cell) for cell in cells[1:])
        anchors.append(goal)
        return self._smooth_path(anchors)

    def _point_to_cell(self, point: Point3) -> Cell:
        return tuple(
            max(
                0,
                min(
                    self._max_index[i],
                    int(round((point[i] - self.bounds_min[i]) / self.resolution)),
                ),
            )
            for i in range(3)
        )

    def _cell_to_point(self, cell: Cell) -> Point3:
        return tuple(
            self.bounds_min[i] + cell[i] * self.resolution for i in range(3)
        )

    def _heuristic(self, a: Cell, b: Cell) -> float:
        return sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))) * self.resolution

    def _step_is_free(self, start: Point3, end: Point3) -> bool:
        """Reject a one-cell step that cuts an inflated obstacle corner anywhere along it."""
        if not self._segment_clear_of_boxes(start, end):
            return False
        if self.is_blocked is None:
            return True
        midpoint = tuple((start[axis] + end[axis]) * 0.5 for axis in range(3))
        return not self.is_blocked(midpoint)

    def _segment_clear_of_boxes(self, start: Point3, end: Point3) -> bool:
        """Exact slab test of the segment against every nearby inflated box."""
        low = [floor(min(start[axis], end[axis]) / _BUCKET_SIZE) for axis in range(2)]
        high = [floor(max(start[axis], end[axis]) / _BUCKET_SIZE) for axis in range(2)]
        seen = set()
        for bucket_x in range(low[0], high[0] + 1):
            for bucket_y in range(low[1], high[1] + 1):
                for obstacle in self._buckets.get((bucket_x, bucket_y), ()):
                    if id(obstacle) in seen:
                        continue
                    seen.add(id(obstacle))
                    if self._segment_hits_box(start, end, obstacle):
                        return False
        return True

    def _segment_hits_box(self, start: Point3, end: Point3, obstacle: BoxObstacle) -> bool:
        enter, leave = 0.0, 1.0
        for axis in range(3):
            half = obstacle.size[axis] / 2.0 + self.clearance
            box_low = obstacle.center[axis] - half
            box_high = obstacle.center[axis] + half
            delta = end[axis] - start[axis]
            if abs(delta) < 1e-12:
                if start[axis] < box_low or start[axis] > box_high:
                    return False
                continue
            first = (box_low - start[axis]) / delta
            second = (box_high - start[axis]) / delta
            if first > second:
                first, second = second, first
            enter = max(enter, first)
            leave = min(leave, second)
            if enter > leave:
                return False
        return True

    def _segment_is_free(self, start: Point3, end: Point3) -> bool:
        if not (self.point_is_free(start) and self.point_is_free(end)):
            return False
        if not self._segment_clear_of_boxes(start, end):
            return False
        if self.is_blocked is None:
            return True
        segment_length = dist(start, end)
        sample_step = min(self.resolution * 0.25, 0.1)
        sample_count = max(1, ceil(segment_length / sample_step))
        for sample in range(sample_count + 1):
            ratio = sample / sample_count
            point = tuple(start[i] + ratio * (end[i] - start[i]) for i in range(3))
            if self.is_blocked(point):
                return False
        return True

    def _smooth_path(self, points: Sequence[Point3]) -> List[Point3]:
        if len(points) <= 2:
            return list(points)
        path = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            candidate = len(points) - 1
            while candidate > anchor + 1 and not self._segment_is_free(
                points[anchor], points[candidate]
            ):
                candidate -= 1
            path.append(points[candidate])
            anchor = candidate
        return path


def parse_waypoints(flat: Sequence[float]) -> List[Point3]:
    """Group a flat parameter list into ENU waypoints."""
    if len(flat) < 3 or len(flat) % 3:
        raise ValueError("inspection_points must contain one or more x/y/z triples")
    return [
        tuple(float(value) for value in flat[index:index + 3])
        for index in range(0, len(flat), 3)
    ]


def point_in_grid_cells(
    point: Point3,
    bounds_min: Point3,
    bounds_max: Point3,
    resolution: float,
    cells: AbstractSet[Cell],
) -> bool:
    """Check whether a point lies in a blocked cell of a bounded voxel grid."""
    if resolution <= 0.0:
        raise ValueError("Grid resolution must be positive")
    if any(
        point[axis] < bounds_min[axis] or point[axis] > bounds_max[axis]
        for axis in range(3)
    ):
        return True
    max_index = tuple(
        int((bounds_max[axis] - bounds_min[axis]) / resolution) - 1
        for axis in range(3)
    )
    cell = tuple(
        min(
            max_index[axis],
            max(
                0,
                int((point[axis] - bounds_min[axis]) // resolution),
            ),
        )
        for axis in range(3)
    )
    return cell in cells


def path_intersects_grid_cells(
    start: Point3,
    path: Sequence[Point3],
    bounds_min: Point3,
    bounds_max: Point3,
    resolution: float,
    blocked_cells: AbstractSet[Cell],
) -> bool:
    """Sample a path densely enough to detect intersection with blocked voxels."""
    if resolution <= 0.0:
        raise ValueError("Grid resolution must be positive")
    previous = start
    sample_step = min(resolution * 0.25, 0.1)
    for endpoint in path:
        segment_length = dist(previous, endpoint)
        sample_count = max(1, ceil(segment_length / sample_step))
        for sample in range(sample_count + 1):
            ratio = sample / sample_count
            point = tuple(
                previous[axis] + ratio * (endpoint[axis] - previous[axis])
                for axis in range(3)
            )
            if point_in_grid_cells(
                point, bounds_min, bounds_max, resolution, blocked_cells
            ):
                return True
        previous = endpoint
    return False


def parse_box_obstacles(flat: Sequence[float]) -> List[BoxObstacle]:
    """Group center/size values into axis-aligned obstacles."""
    if not flat or len(flat) % 6:
        raise ValueError("obstacles must contain one or more center/size groups of six")
    return [
        BoxObstacle(
            tuple(float(value) for value in flat[index:index + 3]),
            tuple(float(value) for value in flat[index + 3:index + 6]),
        )
        for index in range(0, len(flat), 6)
    ]
