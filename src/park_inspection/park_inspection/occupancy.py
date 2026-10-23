"""Probabilistic 3D occupancy grid for a forward depth camera.

The update follows OctoMap: each voxel stores a log-odds occupancy, a hit adds
evidence, and voxels traversed before the return are marked free. Like OctoMap
and ROG-Map, a voxel is updated at most once per scan so repeated rays do not
over-weight the same cell. The grid stays fixed to the park bounds instead of
a rolling vehicle-centered map; the park is small enough that a uniform grid
can be compared directly with the known obstacle boxes.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from park_inspection.planning import (
    BoxObstacle,
    GridPlanner3D,
    Point3,
)

Cell = Tuple[int, int, int]


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be between 0 and 1")
    return math.log(probability / (1.0 - probability))


class OccupancyMap3D:
    """Log-odds occupancy on the same bounds and resolution as the park planner."""

    def __init__(
        self,
        bounds_min: Point3,
        bounds_max: Point3,
        resolution: float = 1.0,
        clearance: float = 1.0,
        prob_hit: float = 0.7,
        prob_miss: float = 0.4,
        clamp_min: float = 0.12,
        clamp_max: float = 0.97,
        occupancy_threshold: float = 0.5,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("Grid resolution must be positive")
        if clearance < 0.0:
            raise ValueError("Obstacle clearance cannot be negative")
        if any(bounds_min[i] >= bounds_max[i] for i in range(3)):
            raise ValueError("Each minimum bound must be less than its maximum")
        if prob_miss >= 0.5 or prob_hit <= 0.5:
            raise ValueError("Hits must raise occupancy and misses must lower it")
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.resolution = resolution
        self.clearance = clearance
        self._hit_log = _logit(prob_hit)
        self._miss_log = _logit(prob_miss)
        self._clamp_min = _logit(clamp_min)
        self._clamp_max = _logit(clamp_max)
        self._threshold = _logit(occupancy_threshold)
        self._max_index = tuple(
            int((bounds_max[i] - bounds_min[i]) / resolution) - 1
            for i in range(3)
        )
        if any(index < 0 for index in self._max_index):
            raise ValueError("Bounds are smaller than one grid cell")
        self._log_odds: Dict[Cell, float] = {}
        self._occupied = set()
        # Number of occupied voxels within clearance of each blocked cell.
        self._blocked_counts: Dict[Cell, int] = {}
        self._blocked_generation = 0
        self._last_updated_cells: Tuple[Cell, ...] = ()
        self._prior_cache: Optional[Tuple[Tuple[BoxObstacle, ...], set]] = None
        radius = int(math.ceil((clearance + resolution) / resolution))
        self._inflation_offsets = tuple(
            (dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
            if self._offset_distance((dx, dy, dz)) <= clearance
        )

    @property
    def blocked_generation(self) -> int:
        """Incremented only when the safety-relevant blocked voxel set changes."""
        return self._blocked_generation

    def integrate_scan(
        self,
        origin: Point3,
        endpoints: Sequence[Point3],
        max_range: float,
        min_range: float = 0.3,
        ground_z: float = 0.4,
    ) -> int:
        """Ray-cast one cloud. Returns the number of accepted returns."""
        if max_range <= 0.0 or min_range < 0.0 or min_range >= max_range:
            raise ValueError("Range limits must satisfy 0 <= min_range < max_range")
        hits = set()
        misses = set()
        accepted = 0
        for endpoint in endpoints:
            clipped = self._clip_return(origin, endpoint, max_range, min_range, ground_z)
            if clipped is None:
                continue
            accepted += 1
            end_point, is_hit = clipped
            cells = self.cells_along(origin, end_point)
            if not cells:
                continue
            if is_hit:
                hits.add(cells[-1])
                misses.update(cells[:-1])
            else:
                misses.update(cells)
        misses.difference_update(hits)
        for cell in hits:
            self._update(cell, self._hit_log)
        for cell in misses:
            self._update(cell, self._miss_log)
        self._last_updated_cells = tuple(hits | misses)
        return accepted

    @property
    def last_updated_cells(self) -> Tuple[Cell, ...]:
        """Cells whose log-odds changed in the most recent scan."""
        return self._last_updated_cells

    def snapshot(self) -> "OccupancyMap3D":
        """Independent copy for planning on another thread while scans continue."""
        copy = object.__new__(OccupancyMap3D)
        copy.__dict__.update(self.__dict__)
        copy._log_odds = dict(self._log_odds)
        copy._occupied = set(self._occupied)
        copy._blocked_counts = dict(self._blocked_counts)
        return copy

    def cell_at(self, point: Point3) -> Optional[Cell]:
        return self._point_to_cell(point)

    def contains_cell(self, cell: Cell) -> bool:
        return all(
            0 <= cell[axis] <= self._max_index[axis] for axis in range(3)
        )

    def is_unknown(self, cell: Cell) -> bool:
        """Unobserved in-bounds cells. Cells outside the map are not unknown."""
        return self.contains_cell(cell) and cell not in self._log_odds

    def is_free(self, cell: Cell) -> bool:
        odds = self._log_odds.get(cell)
        return odds is not None and odds <= self._threshold

    def is_occupied(self, cell: Cell) -> bool:
        odds = self._log_odds.get(cell)
        return odds is not None and odds > self._threshold

    def blocks(self, point: Point3) -> bool:
        """True when ``point`` lies in a voxel inflated by the planning clearance."""
        cell = self._point_to_cell(point)
        if cell is None:
            return True
        return cell in self._blocked_counts

    def occupied_cells(self) -> List[Cell]:
        return list(self._occupied)

    def blocked_cells(self) -> List[Cell]:
        """Return occupied voxels expanded by this map's planning clearance."""
        return sorted(self._blocked_counts)

    def cell_center(self, cell: Cell) -> Point3:
        return tuple(
            self.bounds_min[axis] + (cell[axis] + 0.5) * self.resolution
            for axis in range(3)
        )

    def agreement(self, obstacles: Sequence[BoxObstacle]) -> dict:
        """Compare observed voxels with the known box map.

        Recall uses only prior-occupied voxels the sensor has actually
        observed. Unseen buildings stay out of the false-negative count.
        """
        prior = self._prior_cells(obstacles)
        occupied = set(self.occupied_cells())
        observed = set(self._log_odds)
        true_positive = occupied & prior
        false_positive = occupied - prior
        observed_prior = observed & prior
        false_negative = observed_prior - occupied
        return {
            "occupied_cells": len(occupied),
            "observed_cells": len(observed),
            "prior_occupied_cells": len(prior),
            "true_positive_cells": len(true_positive),
            "false_positive_cells": len(false_positive),
            "false_negative_cells": len(false_negative),
            "precision": _ratio(len(true_positive), len(occupied)),
            "recall_observed": _ratio(len(true_positive), len(observed_prior)),
        }

    def cells_along(self, start: Point3, end: Point3) -> List[Cell]:
        """Visit each voxel of the segment once, in Amanatides–Woo order."""
        start_g = self._grid_coordinate(start)
        end_g = self._grid_coordinate(end)
        current = [
            self._voxel_index(start_g[axis], end_g[axis] - start_g[axis])
            for axis in range(3)
        ]
        last = [int(math.floor(end_g[axis])) for axis in range(3)]
        direction = [end_g[axis] - start_g[axis] for axis in range(3)]
        step = []
        t_max = []
        t_delta = []
        for axis in range(3):
            delta = direction[axis]
            if delta == 0.0:
                step.append(0)
                t_max.append(math.inf)
                t_delta.append(math.inf)
                continue
            step.append(1 if delta > 0.0 else -1)
            if delta > 0.0:
                boundary = math.floor(start_g[axis]) + 1.0
                t_max.append((boundary - start_g[axis]) / delta)
            else:
                boundary = math.floor(start_g[axis])
                t_max.append((start_g[axis] - boundary) / -delta)
            t_delta.append(abs(1.0 / delta))

        cells: List[Cell] = []
        limit = sum(index + 1 for index in self._max_index) + 3
        for _ in range(limit):
            if any(
                current[axis] < 0 or current[axis] > self._max_index[axis]
                for axis in range(3)
            ):
                break
            cells.append((current[0], current[1], current[2]))
            if current == last or min(t_max) > 1.0:
                break
            axis = min(range(3), key=t_max.__getitem__)
            current[axis] += step[axis]
            t_max[axis] += t_delta[axis]
        return cells

    def _clip_return(
        self,
        origin: Point3,
        endpoint: Point3,
        max_range: float,
        min_range: float,
        ground_z: float,
    ) -> Optional[Tuple[Point3, bool]]:
        if not all(math.isfinite(value) for value in endpoint):
            return None
        delta = tuple(endpoint[axis] - origin[axis] for axis in range(3))
        length = math.sqrt(sum(value * value for value in delta))
        if not math.isfinite(length) or length < min_range:
            return None
        is_hit = True
        if length > max_range:
            scale = max_range / length
            endpoint = tuple(origin[axis] + delta[axis] * scale for axis in range(3))
            is_hit = False
        elif endpoint[2] < ground_z:
            is_hit = False
        return endpoint, is_hit

    def _update(self, cell: Cell, delta: float) -> None:
        updated = self._log_odds.get(cell, 0.0) + delta
        updated = min(self._clamp_max, max(self._clamp_min, updated))
        self._log_odds[cell] = updated
        occupied = updated > self._threshold
        if occupied == (cell in self._occupied):
            return
        if occupied:
            self._occupied.add(cell)
        else:
            self._occupied.discard(cell)
        step = 1 if occupied else -1
        changed = False
        for dx, dy, dz in self._inflation_offsets:
            neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if not self.contains_cell(neighbor):
                continue
            count = self._blocked_counts.get(neighbor, 0) + step
            if count > 0:
                if count == 1 and step > 0:
                    changed = True
                self._blocked_counts[neighbor] = count
            else:
                del self._blocked_counts[neighbor]
                changed = True
        if changed:
            self._blocked_generation += 1

    def _offset_distance(self, offset: Cell) -> float:
        """Distance from a neighbor cell center to the occupied voxel at the origin."""
        half = self.resolution / 2.0
        outside = [max(abs(value) * self.resolution - half, 0.0) for value in offset]
        if any(value > 0.0 for value in outside):
            return math.sqrt(sum(value * value for value in outside))
        return -half

    def _grid_coordinate(self, point: Point3) -> Point3:
        return tuple(
            (point[axis] - self.bounds_min[axis]) / self.resolution
            for axis in range(3)
        )

    def _voxel_index(self, coordinate: float, direction: float) -> int:
        if direction < 0.0 and math.isclose(coordinate, round(coordinate), abs_tol=1e-9):
            return int(round(coordinate)) - 1
        return int(math.floor(coordinate))

    def _point_to_cell(self, point: Point3) -> Optional[Cell]:
        if any(
            point[axis] < self.bounds_min[axis] or point[axis] > self.bounds_max[axis]
            for axis in range(3)
        ):
            return None
        cell = tuple(
            min(
                self._max_index[axis],
                max(0, int(math.floor(
                    (point[axis] - self.bounds_min[axis]) / self.resolution
                ))),
            )
            for axis in range(3)
        )
        return cell

    def _prior_cells(self, obstacles: Sequence[BoxObstacle]) -> set:
        """Voxels whose centers lie inside a known box, cached per obstacle list."""
        key = tuple(obstacles)
        if self._prior_cache is not None and self._prior_cache[0] == key:
            return self._prior_cache[1]
        cells = set()
        for obstacle in key:
            ranges = []
            for axis in range(3):
                low = obstacle.center[axis] - obstacle.size[axis] / 2.0
                high = obstacle.center[axis] + obstacle.size[axis] / 2.0
                first = math.ceil((low - self.bounds_min[axis]) / self.resolution - 0.5)
                last = math.floor((high - self.bounds_min[axis]) / self.resolution - 0.5)
                ranges.append(
                    range(max(0, first), min(self._max_index[axis], last) + 1)
                )
            for ix in ranges[0]:
                for iy in ranges[1]:
                    for iz in ranges[2]:
                        if obstacle.contains(self.cell_center((ix, iy, iz))):
                            cells.add((ix, iy, iz))
        self._prior_cache = (key, cells)
        return cells


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 3)


# The prior route depends only on the known boxes, so it is planned once per
# geometry and waypoint list rather than on every map comparison.
_PRIOR_ROUTES: Dict[tuple, Optional[dict]] = {}


def _path_length(path: Sequence[Point3]) -> float:
    return sum(math.dist(start, end) for start, end in zip(path, path[1:]))


def _plan_route(planner: GridPlanner3D, waypoints: Sequence[Point3]) -> Optional[dict]:
    if len(waypoints) < 2:
        raise ValueError("A route needs at least two waypoints")
    length = 0.0
    points: List[Point3] = [waypoints[0]]
    try:
        for start, goal in zip(waypoints, waypoints[1:]):
            segment = planner.plan(start, goal)
            length += _path_length(segment)
            points.extend(segment[1:])
    except (ValueError, RuntimeError):
        return None
    return {"length_m": round(length, 2), "points": points}


def _prior_intersections(
    path: Sequence[Point3],
    obstacles: Sequence[BoxObstacle],
    sample_step: float = 0.5,
) -> int:
    violations = 0
    for start, end in zip(path, path[1:]):
        segment_length = math.dist(start, end)
        sample_count = max(1, int(math.ceil(segment_length / sample_step)))
        for sample in range(sample_count + 1):
            ratio = sample / sample_count
            point = tuple(start[axis] + ratio * (end[axis] - start[axis]) for axis in range(3))
            if any(obstacle.contains(point) for obstacle in obstacles):
                violations += 1
    return violations


def compare_inspection_plans(
    bounds_min: Point3,
    bounds_max: Point3,
    obstacles: Sequence[BoxObstacle],
    resolution: float,
    clearance: float,
    occupancy: OccupancyMap3D,
    waypoints: Sequence[Point3],
) -> dict:
    """Plan the same route on the prior map, the sensor map, and their union.

    Unknown voxels stay free. The prior route still avoids the known boxes,
    and the fused route also avoids voxels the depth camera has marked
    occupied. The online route ignores the prior boxes, so its prior-intersection
    count shows where perception alone would cut through an unseen building.
    """
    common = {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "resolution": resolution,
        "clearance": clearance,
    }
    prior_key = (
        tuple(bounds_min),
        tuple(bounds_max),
        tuple(obstacles),
        resolution,
        clearance,
        tuple(tuple(point) for point in waypoints),
    )
    if prior_key not in _PRIOR_ROUTES:
        if len(_PRIOR_ROUTES) >= 8:
            _PRIOR_ROUTES.clear()
        _PRIOR_ROUTES[prior_key] = _plan_route(
            GridPlanner3D(obstacles=obstacles, **common), waypoints
        )
    prior = _PRIOR_ROUTES[prior_key]
    online = _plan_route(
        GridPlanner3D(obstacles=[], is_blocked=occupancy.blocks, **common),
        waypoints,
    )
    fused = _plan_route(
        GridPlanner3D(obstacles=obstacles, is_blocked=occupancy.blocks, **common),
        waypoints,
    )
    result = {
        "prior_plan_ok": prior is not None,
        "online_plan_ok": online is not None,
        "fused_plan_ok": fused is not None,
        "prior_path_length_m": None if prior is None else prior["length_m"],
        "online_path_length_m": None if online is None else online["length_m"],
        "fused_path_length_m": None if fused is None else fused["length_m"],
        "online_path_prior_intersections": (
            None
            if online is None
            else _prior_intersections(online["points"], obstacles)
        ),
    }
    result.update(occupancy.agreement(obstacles))
    return result
