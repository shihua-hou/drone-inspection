"""Frontier exploration adapted from FUEL for the park occupancy grid.

FUEL keeps a frontier information structure and plans in three coarse-to-fine
steps: a global tour of frontier clusters, a local choice among viewpoints,
then a minimum-time trajectory. This module keeps the first two steps on the
existing voxel map. Large clusters are split on their longest grid axis rather
than by principal-component analysis. Connection costs use a straight-line
time lower bound, and the open tour is nearest-neighbor plus 2-opt rather than
an LKH solver. The local path is the grid A* already used for inspection, not
a B-spline. The inspection Offboard route is left unchanged; callers publish
or follow this plan separately.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from park_inspection.occupancy import OccupancyMap3D
from park_inspection.planning import GridPlanner3D, Point3

Cell = Tuple[int, int, int]
_NEIGHBORS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
)


@dataclass(frozen=True)
class Viewpoint:
    position: Point3
    yaw: float
    coverage: int


@dataclass(frozen=True)
class FrontierCluster:
    cells: Tuple[Cell, ...]
    average: Point3
    bounds_min: Cell
    bounds_max: Cell
    viewpoints: Tuple[Viewpoint, ...]


@dataclass(frozen=True)
class ExplorationPlan:
    """Global visit order, the refined local prefix, and the path to its first point."""

    tour: Tuple[Viewpoint, ...]
    local_viewpoints: Tuple[Viewpoint, ...]
    path: Tuple[Point3, ...]


class FrontierInformationStructure:
    """Incremental frontier clusters for the cells a scan just changed."""

    def __init__(
        self,
        min_cluster_cells: int = 2,
        max_cluster_cells: int = 24,
        viewpoint_radius: float = 4.0,
        viewpoint_count: int = 8,
        max_viewpoints: int = 4,
        sensor_range: float = 15.0,
        half_hfov: float = 0.65,
    ) -> None:
        if min_cluster_cells < 1 or max_cluster_cells < min_cluster_cells:
            raise ValueError("Cluster size limits are inconsistent")
        if viewpoint_radius <= 0.0 or viewpoint_count < 4 or max_viewpoints < 1:
            raise ValueError("Viewpoint sampling parameters are invalid")
        if sensor_range <= 0.0 or half_hfov <= 0.0:
            raise ValueError("Sensor coverage parameters must be positive")
        self.min_cluster_cells = min_cluster_cells
        self.max_cluster_cells = max_cluster_cells
        self.viewpoint_radius = viewpoint_radius
        self.viewpoint_count = viewpoint_count
        self.max_viewpoints = max_viewpoints
        self.sensor_range = sensor_range
        self.half_hfov = half_hfov
        self._clusters: List[FrontierCluster] = []

    @property
    def clusters(self) -> Tuple[FrontierCluster, ...]:
        return tuple(self._clusters)

    def update(
        self, occupancy: OccupancyMap3D, updated_cells: Sequence[Cell]
    ) -> bool:
        """Drop stale clusters that meet this scan and grow frontiers beside it."""
        if not updated_cells:
            return False
        box_min, box_max = _cell_bounds(updated_cells, padding=1)
        kept: List[FrontierCluster] = []
        changed = False
        for cluster in self._clusters:
            if not _boxes_overlap(
                cluster.bounds_min, cluster.bounds_max, box_min, box_max
            ):
                kept.append(cluster)
                continue
            if any(not _is_frontier(occupancy, cell) for cell in cluster.cells):
                changed = True
                continue
            kept.append(cluster)
        claimed = {cell for cluster in kept for cell in cluster.cells}
        fresh: List[FrontierCluster] = []
        for ix in range(box_min[0], box_max[0] + 1):
            for iy in range(box_min[1], box_max[1] + 1):
                for iz in range(box_min[2], box_max[2] + 1):
                    seed = (ix, iy, iz)
                    if seed in claimed or not _is_frontier(occupancy, seed):
                        continue
                    cells = _grow(occupancy, seed, claimed)
                    for piece in _split(cells, self.max_cluster_cells):
                        if len(piece) < self.min_cluster_cells:
                            continue
                        cluster = self._make_cluster(occupancy, piece)
                        if cluster.viewpoints:
                            fresh.append(cluster)
                            changed = True
        if not changed and len(kept) == len(self._clusters):
            return False
        self._clusters = kept + fresh
        return True

    def _make_cluster(
        self, occupancy: OccupancyMap3D, cells: Sequence[Cell]
    ) -> FrontierCluster:
        average = tuple(
            sum(occupancy.cell_center(cell)[axis] for cell in cells) / len(cells)
            for axis in range(3)
        )
        bounds_min = tuple(min(cell[axis] for cell in cells) for axis in range(3))
        bounds_max = tuple(max(cell[axis] for cell in cells) for axis in range(3))
        return FrontierCluster(
            cells=tuple(cells),
            average=average,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            viewpoints=self._viewpoints(occupancy, cells, average),
        )

    def _viewpoints(
        self,
        occupancy: OccupancyMap3D,
        cells: Sequence[Cell],
        average: Point3,
    ) -> Tuple[Viewpoint, ...]:
        found = []
        for index in range(self.viewpoint_count):
            angle = -math.pi + (2.0 * math.pi * index) / self.viewpoint_count
            position = (
                average[0] + self.viewpoint_radius * math.cos(angle),
                average[1] + self.viewpoint_radius * math.sin(angle),
                average[2],
            )
            cell = occupancy.cell_at(position)
            if cell is None or not occupancy.is_free(cell) or occupancy.blocks(position):
                continue
            yaw = math.atan2(average[1] - position[1], average[0] - position[0])
            coverage = _coverage(
                occupancy, position, yaw, cells, self.sensor_range, self.half_hfov
            )
            if coverage <= 0:
                continue
            found.append(Viewpoint(position=position, yaw=yaw, coverage=coverage))
        found.sort(key=lambda viewpoint: viewpoint.coverage, reverse=True)
        return tuple(found[: self.max_viewpoints])


def plan_exploration(
    start: Point3,
    yaw: float,
    clusters: Sequence[FrontierCluster],
    planner: GridPlanner3D,
    velocity: Point3 = (0.0, 0.0, 0.0),
    v_max: float = 3.0,
    yaw_rate_max: float = 1.0,
    consistency_weight: float = 1.5,
    refine_radius: float = 15.0,
) -> Optional[ExplorationPlan]:
    """Plan an open cluster tour and an A* path to its first refined viewpoint."""
    if v_max <= 0.0 or yaw_rate_max <= 0.0:
        raise ValueError("Speed limits must be positive")
    usable = [cluster for cluster in clusters if cluster.viewpoints]
    if not usable or not planner.point_is_free(start):
        return None
    ordered = _two_opt(
        start,
        yaw,
        _nearest_neighbor(
            start, yaw, velocity, usable, v_max, yaw_rate_max, consistency_weight
        ),
        v_max,
        yaw_rate_max,
    )
    prefix = [ordered[0]]
    for cluster in ordered[1:]:
        if math.dist(start, cluster.viewpoints[0].position) > refine_radius:
            break
        prefix.append(cluster)
        if len(prefix) >= 3:
            break
    local = _refine_prefix(
        start, yaw, velocity, prefix, v_max, yaw_rate_max, consistency_weight
    )
    goal_order = list(local) + [
        cluster.viewpoints[0] for cluster in ordered[len(local):]
    ]
    path = _path_to_first_viewpoint(start, prefix[0], local[0], planner)
    if path is None:
        return None
    return ExplorationPlan(
        tour=tuple(goal_order),
        local_viewpoints=tuple(local),
        path=tuple(path),
    )


def _is_frontier(occupancy: OccupancyMap3D, cell: Cell) -> bool:
    if not occupancy.is_free(cell):
        return False
    for offset in _NEIGHBORS:
        neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
        if occupancy.is_unknown(neighbor):
            return True
    return False


def _grow(
    occupancy: OccupancyMap3D, seed: Cell, claimed: set
) -> List[Cell]:
    cells = []
    stack = [seed]
    seen = {seed}
    while stack:
        cell = stack.pop()
        if cell in claimed or not _is_frontier(occupancy, cell):
            continue
        claimed.add(cell)
        cells.append(cell)
        for offset in _NEIGHBORS:
            neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
            if neighbor in seen or not occupancy.contains_cell(neighbor):
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return cells


def _split(cells: Sequence[Cell], max_cells: int) -> List[List[Cell]]:
    if len(cells) <= max_cells:
        return [list(cells)]
    mean = [sum(cell[axis] for cell in cells) / len(cells) for axis in range(3)]
    variance = [
        sum((cell[axis] - mean[axis]) ** 2 for cell in cells) for axis in range(3)
    ]
    axis = max(range(3), key=variance.__getitem__)
    if variance[axis] <= 0.0:
        return [list(cells)]
    left = [cell for cell in cells if cell[axis] <= mean[axis]]
    right = [cell for cell in cells if cell[axis] > mean[axis]]
    if not left or not right:
        return [list(cells)]
    return _split(left, max_cells) + _split(right, max_cells)


def _coverage(
    occupancy: OccupancyMap3D,
    origin: Point3,
    yaw: float,
    cells: Sequence[Cell],
    sensor_range: float,
    half_hfov: float,
) -> int:
    covered = 0
    for cell in cells[:32]:
        center = occupancy.cell_center(cell)
        delta = tuple(center[axis] - origin[axis] for axis in range(3))
        distance = math.sqrt(sum(value * value for value in delta))
        if distance > sensor_range or distance < 1e-6:
            continue
        bearing = math.atan2(delta[1], delta[0])
        if _yaw_delta(yaw, bearing) > half_hfov:
            continue
        if _occluded(occupancy, origin, center, cell):
            continue
        covered += 1
    return covered


def _occluded(
    occupancy: OccupancyMap3D, origin: Point3, end: Point3, target: Cell
) -> bool:
    for cell in occupancy.cells_along(origin, end)[:-1]:
        if cell == target:
            break
        if occupancy.is_occupied(cell):
            return True
    return False


def _nearest_neighbor(
    start: Point3,
    yaw: float,
    velocity: Point3,
    clusters: Sequence[FrontierCluster],
    v_max: float,
    yaw_rate_max: float,
    consistency_weight: float,
) -> List[FrontierCluster]:
    remaining = list(clusters)
    ordered = []
    position = start
    current_yaw = yaw
    while remaining:
        def cost(cluster: FrontierCluster, first: bool = not ordered) -> float:
            viewpoint = cluster.viewpoints[0]
            value = _time_lower_bound(
                position, current_yaw, viewpoint, v_max, yaw_rate_max
            )
            if first:
                value += consistency_weight * _consistency(
                    position, viewpoint.position, velocity
                )
            return value

        chosen = min(remaining, key=cost)
        ordered.append(chosen)
        remaining.remove(chosen)
        position = chosen.viewpoints[0].position
        current_yaw = chosen.viewpoints[0].yaw
    return ordered


def _two_opt(
    start: Point3,
    yaw: float,
    clusters: Sequence[FrontierCluster],
    v_max: float,
    yaw_rate_max: float,
) -> List[FrontierCluster]:
    best = list(clusters)
    if len(best) < 3:
        return best

    def length(order: Sequence[FrontierCluster]) -> float:
        total = 0.0
        position = start
        current_yaw = yaw
        for cluster in order:
            viewpoint = cluster.viewpoints[0]
            total += _time_lower_bound(
                position, current_yaw, viewpoint, v_max, yaw_rate_max
            )
            position = viewpoint.position
            current_yaw = viewpoint.yaw
        return total

    current = length(best)
    for _ in range(8):
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 1, len(best)):
                candidate = best[:left] + list(reversed(best[left:right + 1])) + best[right + 1:]
                candidate_length = length(candidate)
                if candidate_length + 1e-9 < current:
                    best = candidate
                    current = candidate_length
                    improved = True
        if not improved:
            break
    return best


def _refine_prefix(
    start: Point3,
    yaw: float,
    velocity: Point3,
    prefix: Sequence[FrontierCluster],
    v_max: float,
    yaw_rate_max: float,
    consistency_weight: float,
) -> List[Viewpoint]:
    layers = [cluster.viewpoints for cluster in prefix]
    costs = []
    choice = []
    first_costs = []
    for viewpoint in layers[0]:
        value = _time_lower_bound(start, yaw, viewpoint, v_max, yaw_rate_max)
        value += consistency_weight * _consistency(start, viewpoint.position, velocity)
        first_costs.append(value)
    costs.append(first_costs)
    for layer_index in range(1, len(layers)):
        layer_costs = []
        layer_choice = []
        for viewpoint in layers[layer_index]:
            best_cost = math.inf
            best_prev = 0
            for prev_index, previous in enumerate(layers[layer_index - 1]):
                value = costs[-1][prev_index] + _time_lower_bound(
                    previous.position, previous.yaw, viewpoint, v_max, yaw_rate_max
                )
                if value < best_cost:
                    best_cost = value
                    best_prev = prev_index
            layer_costs.append(best_cost)
            layer_choice.append(best_prev)
        costs.append(layer_costs)
        choice.append(layer_choice)
    index = min(range(len(layers[-1])), key=lambda item: costs[-1][item])
    picked = [index]
    for layer_index in range(len(layers) - 1, 0, -1):
        index = choice[layer_index - 1][index]
        picked.append(index)
    picked.reverse()
    return [layers[layer][picked[layer]] for layer in range(len(layers))]


def _path_to_first_viewpoint(
    start: Point3,
    cluster: FrontierCluster,
    preferred: Viewpoint,
    planner: GridPlanner3D,
) -> Optional[List[Point3]]:
    candidates = [preferred] + [
        viewpoint
        for viewpoint in cluster.viewpoints
        if viewpoint.position != preferred.position
    ]
    for viewpoint in candidates:
        try:
            return planner.plan(start, viewpoint.position)
        except (ValueError, RuntimeError):
            continue
    return None


def _time_lower_bound(
    position: Point3,
    yaw: float,
    viewpoint: Viewpoint,
    v_max: float,
    yaw_rate_max: float,
) -> float:
    return max(
        math.dist(position, viewpoint.position) / v_max,
        _yaw_delta(yaw, viewpoint.yaw) / yaw_rate_max,
    )


def _consistency(origin: Point3, target: Point3, velocity: Point3) -> float:
    speed = math.sqrt(sum(value * value for value in velocity))
    if speed < 0.2:
        return 0.0
    direction = tuple(target[axis] - origin[axis] for axis in range(3))
    length = math.sqrt(sum(value * value for value in direction))
    if length < 1e-6:
        return 0.0
    cosine = sum(direction[axis] * velocity[axis] for axis in range(3)) / (length * speed)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _yaw_delta(start: float, end: float) -> float:
    delta = abs(start - end) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def _cell_bounds(cells: Sequence[Cell], padding: int) -> Tuple[Cell, Cell]:
    lower = tuple(min(cell[axis] for cell in cells) - padding for axis in range(3))
    upper = tuple(max(cell[axis] for cell in cells) + padding for axis in range(3))
    return lower, upper


def _boxes_overlap(
    first_min: Cell, first_max: Cell, second_min: Cell, second_max: Cell
) -> bool:
    return all(
        first_max[axis] >= second_min[axis] and first_min[axis] <= second_max[axis]
        for axis in range(3)
    )
