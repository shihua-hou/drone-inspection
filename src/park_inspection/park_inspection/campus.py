"""Campus layout shared by the Gazebo world, the planner map, and the tests.

The inner campus is a regular street grid. Each block holds buildings of mixed
footprint and height with at least 8 m between facades, so the planner,
inflated by its 2.5 m clearance, can still route between them. A belt between
the ring road and the perimeter fence holds a parking lot, a sports field,
service buildings, and a small park. Street trees, lights, signs, and hedges
line the sidewalks.

``tools/generate_campus.py`` writes ``worlds/park.sdf``, the meshes under
``worlds/meshes``, and ``config/campus_obstacles.yaml`` from this module; edit
the layout here, not in the generated files. Every object that a sensor can
see above the ground threshold has a planner box here.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from park_inspection.planning import BoxObstacle, Point3

Rect = Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

STREET_X = (-20.0, 20.0, 60.0)
STREET_Y = (-20.0, 20.0)
RING_X = (-60.0, 100.0)
RING_Y = (-60.0, 60.0)
BLOCK_X = (-40.0, 0.0, 40.0, 80.0)
BLOCK_Y = (-40.0, 0.0, 40.0)
ROAD_HALF_WIDTH = 4.0
SIDEWALK_OUTER = 7.0
BLOCK_HALF = 13.0
PAVEMENT_HALF = BLOCK_HALF + SIDEWALK_OUTER - ROAD_HALF_WIDTH
TREE_OFFSET = 5.0
LIGHT_OFFSET = 4.5
HEDGE_OFFSET = 6.3

FENCE_MIN = (-95.0, -95.0)
FENCE_MAX = (135.0, 95.0)
FENCE_HEIGHT = 2.4
GATE_X = 20.0
GATE_HALF_OPENING = 6.0
PUBLIC_ROAD_Y = -108.0

PARKING_LOT: Rect = (-50.0, -90.0, 15.9, -68.0)
SPORTS_FIELD: Rect = (-10.0, 68.0, 50.0, 92.0)

BOUNDS_MIN: Point3 = (-100.0, -100.0, -1.0)
BOUNDS_MAX: Point3 = (140.0, 100.0, 40.0)
RETURN_ALTITUDE = 6.0

INSPECTION_POINTS: Tuple[Point3, ...] = (
    (40.0, 0.0, 8.0),
    (80.0, 0.0, 10.0),
    (33.0, -37.0, 12.0),
    (-40.0, -40.0, 10.0),
    (0.0, 40.0, 7.0),
)

FACE_NORMALS = {"s": (0.0, -1.0), "n": (0.0, 1.0), "w": (-1.0, 0.0), "e": (1.0, 0.0)}
CANOPY_DEPTH = 1.8
CANOPY_HEIGHT = 3.6


def box_from_bounds(low: Point3, high: Point3) -> BoxObstacle:
    return BoxObstacle(
        center=tuple((low[axis] + high[axis]) / 2.0 for axis in range(3)),
        size=tuple(high[axis] - low[axis] for axis in range(3)),
    )


def _rect_of(box: BoxObstacle, margin: float = 0.0) -> Rect:
    return (
        box.center[0] - box.size[0] / 2.0 - margin,
        box.center[1] - box.size[1] / 2.0 - margin,
        box.center[0] + box.size[0] / 2.0 + margin,
        box.center[1] + box.size[1] / 2.0 + margin,
    )


def rects_overlap(first: Rect, second: Rect) -> bool:
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


@dataclass(frozen=True)
class Building:
    name: str
    center_xy: Tuple[float, float]
    size_xy: Tuple[float, float]
    height: float
    palette: int
    entrance: str
    penthouse: bool = False
    hall: bool = False

    @property
    def curtain_wall(self) -> bool:
        return not self.hall and self.height >= 18.0

    @property
    def obstacle(self) -> BoxObstacle:
        return BoxObstacle(
            center=(self.center_xy[0], self.center_xy[1], self.height / 2.0),
            size=(self.size_xy[0], self.size_xy[1], self.height),
        )

    @property
    def penthouse_obstacle(self) -> Optional[BoxObstacle]:
        if not self.penthouse:
            return None
        width = min(5.0, self.size_xy[0] * 0.4)
        depth = min(5.0, self.size_xy[1] * 0.4)
        return BoxObstacle(
            center=(self.center_xy[0], self.center_xy[1], self.height + 1.5),
            size=(width, depth, 3.0),
        )

    @property
    def entrance_width(self) -> float:
        along_x = self.entrance in ("s", "n")
        face = self.size_xy[0] if along_x else self.size_xy[1]
        return min(6.0, face - 2.0)

    @property
    def canopy_obstacle(self) -> BoxObstacle:
        """Entrance canopy and its columns, protruding from the entrance face."""
        normal = FACE_NORMALS[self.entrance]
        half_x, half_y = self.size_xy[0] / 2.0, self.size_xy[1] / 2.0
        width = self.entrance_width
        if normal[0] == 0.0:
            y_face = self.center_xy[1] + normal[1] * half_y
            y_out = y_face + normal[1] * CANOPY_DEPTH
            return box_from_bounds(
                (self.center_xy[0] - width / 2.0, min(y_face, y_out), 0.0),
                (self.center_xy[0] + width / 2.0, max(y_face, y_out), CANOPY_HEIGHT),
            )
        x_face = self.center_xy[0] + normal[0] * half_x
        x_out = x_face + normal[0] * CANOPY_DEPTH
        return box_from_bounds(
            (min(x_face, x_out), self.center_xy[1] - width / 2.0, 0.0),
            (max(x_face, x_out), self.center_xy[1] + width / 2.0, CANOPY_HEIGHT),
        )


@dataclass(frozen=True)
class Tree:
    """A broadleaf street or park tree. ``variant`` selects a generated mesh."""

    name: str
    position_xy: Tuple[float, float]
    variant: int
    scale: float
    yaw: float

    @property
    def radius(self) -> float:
        return TREE_VARIANTS[self.variant]["radius"] * self.scale

    @property
    def top(self) -> float:
        return TREE_VARIANTS[self.variant]["top"] * self.scale

    @property
    def canopy_center_z(self) -> float:
        return TREE_VARIANTS[self.variant]["canopy_z"] * self.scale

    @property
    def canopy_radius(self) -> float:
        return TREE_VARIANTS[self.variant]["canopy_radius"] * self.scale

    @property
    def obstacle(self) -> BoxObstacle:
        return BoxObstacle(
            center=(self.position_xy[0], self.position_xy[1], self.top / 2.0),
            size=(2.0 * self.radius, 2.0 * self.radius, self.top),
        )


# Bounds of the generated tree meshes before scaling. The generator asserts
# that every mesh vertex lies inside these.
TREE_VARIANTS = (
    {"radius": 1.9, "top": 6.3, "canopy_z": 4.4, "canopy_radius": 1.6},
    {"radius": 1.6, "top": 7.2, "canopy_z": 5.0, "canopy_radius": 1.4},
    {"radius": 2.2, "top": 5.6, "canopy_z": 3.8, "canopy_radius": 1.8},
)


@dataclass(frozen=True)
class StreetLight:
    """Pole at ``position_xy``; the arm reaches ``LIGHT_ARM`` towards ``facing``."""

    name: str
    position_xy: Tuple[float, float]
    facing: Tuple[float, float]

    @property
    def obstacle(self) -> BoxObstacle:
        x_value, y_value = self.position_xy
        arm_x = x_value + self.facing[0] * LIGHT_ARM
        arm_y = y_value + self.facing[1] * LIGHT_ARM
        return box_from_bounds(
            (min(x_value, arm_x) - 0.3, min(y_value, arm_y) - 0.3, 0.0),
            (max(x_value, arm_x) + 0.3, max(y_value, arm_y) + 0.3, LIGHT_HEIGHT + 0.3),
        )


LIGHT_HEIGHT = 8.0
LIGHT_ARM = 1.6


@dataclass(frozen=True)
class Sign:
    name: str
    position_xy: Tuple[float, float]
    yaw: float
    color: int

    @property
    def obstacle(self) -> BoxObstacle:
        return BoxObstacle(
            center=(self.position_xy[0], self.position_xy[1], SIGN_HEIGHT / 2.0),
            size=(1.0, 1.0, SIGN_HEIGHT),
        )


SIGN_HEIGHT = 2.9


@dataclass(frozen=True)
class Hedge:
    name: str
    rect: Rect

    @property
    def obstacle(self) -> BoxObstacle:
        return box_from_bounds(
            (self.rect[0], self.rect[1], 0.0), (self.rect[2], self.rect[3], HEDGE_HEIGHT)
        )


HEDGE_HEIGHT = 0.9


@dataclass(frozen=True)
class Car:
    name: str
    position_xy: Tuple[float, float]
    yaw: float
    color: int

    @property
    def obstacle(self) -> BoxObstacle:
        along_x = abs(math.cos(self.yaw)) > 0.5
        size = (CAR_LENGTH, CAR_WIDTH) if along_x else (CAR_WIDTH, CAR_LENGTH)
        return BoxObstacle(
            center=(self.position_xy[0], self.position_xy[1], CAR_HEIGHT / 2.0),
            size=(size[0] + 0.1, size[1] + 0.1, CAR_HEIGHT),
        )


CAR_LENGTH = 4.4
CAR_WIDTH = 1.8
CAR_HEIGHT = 1.5


@dataclass(frozen=True)
class Structure:
    """Static box-shaped item: fence runs, gate pillars, booth, barrier."""

    name: str
    box: BoxObstacle
    kind: str


def _towers(prefix, bx, by, heights, palettes):
    offsets = ((-8.5, -8.5), (8.5, -8.5), (-8.5, 8.5), (8.5, 8.5))
    return [
        Building(
            "%s_%d" % (prefix, index + 1),
            (bx + dx, by + dy),
            (9.0, 9.0),
            heights[index],
            palettes[index],
            "s" if dy < 0.0 else "n",
            penthouse=heights[index] >= 20.0,
        )
        for index, (dx, dy) in enumerate(offsets)
    ]


def _slabs(prefix, bx, by, heights, palettes):
    return [
        Building("%s_s" % prefix, (bx, by - 8.5), (26.0, 9.0), heights[0], palettes[0], "s",
                 penthouse=heights[0] >= 20.0),
        Building("%s_n" % prefix, (bx, by + 8.5), (26.0, 9.0), heights[1], palettes[1], "n",
                 penthouse=heights[1] >= 20.0),
    ]


def _tower_and_wing(prefix, bx, by, tower_height, wing_height, palettes):
    return [
        Building("%s_tower" % prefix, (bx - 7.0, by - 7.0), (12.0, 12.0), tower_height,
                 palettes[0], "s", penthouse=True),
        Building("%s_wing" % prefix, (bx, by + 10.0), (26.0, 6.0), wing_height, palettes[1], "n"),
    ]


def _podium(prefix, bx, by, height, palette, penthouse=False):
    return [Building(prefix, (bx, by), (20.0, 20.0), height, palette, "s", penthouse=penthouse)]


BUILDINGS: Tuple[Building, ...] = tuple(
    _towers("sw_towers", -40.0, -40.0, (12.0, 18.0, 15.0, 24.0), (0, 1, 2, 3))
    + _slabs("s_offices", 0.0, -40.0, (10.0, 14.0), (4, 5))
    + _tower_and_wing("se_hq", 40.0, -40.0, 30.0, 8.0, (6, 0))
    + _towers("far_se_labs", 80.0, -40.0, (9.0, 12.0, 12.0, 9.0), (5, 2, 1, 4))
    + _podium("w_library", -40.0, 0.0, 12.0, 3)
    + _towers("c_towers", 40.0, 0.0, (16.0, 26.0, 20.0, 14.0), (1, 6, 0, 2))
    + _slabs("e_slabs", 80.0, 0.0, (18.0, 22.0), (3, 5))
    + _tower_and_wing("nw_hotel", -40.0, 40.0, 28.0, 10.0, (2, 4))
    + _towers("n_dorms", 0.0, 40.0, (14.0, 10.0, 20.0, 16.0), (4, 3, 6, 1))
    + _podium("ne_hall", 40.0, 40.0, 9.0, 5, penthouse=True)
    + _towers("far_ne_studios", 80.0, 40.0, (8.0, 11.0, 13.0, 7.0), (0, 2, 4, 6))
    + [
        Building("n_gym", (-40.0, 80.0), (30.0, 20.0), 12.0, 2, "s", hall=True),
        Building("n_canteen", (80.0, 78.0), (20.0, 16.0), 6.0, 1, "s"),
        Building("e_warehouse", (120.0, 10.0), (18.0, 50.0), 11.0, 5, "w", hall=True),
        Building("s_visitor_center", (55.0, -80.0), (16.0, 12.0), 7.0, 4, "w"),
    ]
)


def _structures() -> List[Structure]:
    items = []
    low_x, low_y = FENCE_MIN
    high_x, high_y = FENCE_MAX
    thickness = 0.5
    runs = (
        ("fence_south_w", (low_x, low_y - thickness / 2.0), (GATE_X - GATE_HALF_OPENING, low_y + thickness / 2.0)),
        ("fence_south_e", (GATE_X + GATE_HALF_OPENING, low_y - thickness / 2.0), (high_x, low_y + thickness / 2.0)),
        ("fence_north", (low_x, high_y - thickness / 2.0), (high_x, high_y + thickness / 2.0)),
        ("fence_west", (low_x - thickness / 2.0, low_y), (low_x + thickness / 2.0, high_y)),
        ("fence_east", (high_x - thickness / 2.0, low_y), (high_x + thickness / 2.0, high_y)),
    )
    for name, low, high in runs:
        items.append(
            Structure(name, box_from_bounds((low[0], low[1], 0.0), (high[0], high[1], FENCE_HEIGHT)), "fence")
        )
    for side, x_value in (("w", GATE_X - GATE_HALF_OPENING - 0.6), ("e", GATE_X + GATE_HALF_OPENING + 0.6)):
        items.append(
            Structure(
                "gate_pillar_" + side,
                box_from_bounds((x_value - 0.6, low_y - 0.6, 0.0), (x_value + 0.6, low_y + 0.6, 6.4)),
                "pillar",
            )
        )
    items.append(
        Structure(
            "gate_header",
            box_from_bounds(
                (GATE_X - GATE_HALF_OPENING - 1.2, low_y - 0.5, 5.4),
                (GATE_X + GATE_HALF_OPENING + 1.2, low_y + 0.5, 6.4),
            ),
            "header",
        )
    )
    items.append(
        Structure(
            "gate_booth",
            box_from_bounds((27.5, -91.5, 0.0), (31.5, -87.5, 3.2)),
            "booth",
        )
    )
    items.append(
        Structure(
            "gate_barrier",
            box_from_bounds((GATE_X - 0.2, -88.6, 0.0), (GATE_X + ROAD_HALF_WIDTH + 0.3, -88.0, 1.2)),
            "barrier",
        )
    )
    items.append(
        Structure(
            "gate_leaf",
            box_from_bounds(
                (GATE_X + GATE_HALF_OPENING + 1.4, low_y + 0.3, 0.0),
                (GATE_X + GATE_HALF_OPENING + 1.4 + 2 * GATE_HALF_OPENING, low_y + 0.6, 2.2),
            ),
            "leaf",
        )
    )
    field_y = (SPORTS_FIELD[1] + SPORTS_FIELD[3]) / 2.0
    for name, x_low in (("goal_w", SPORTS_FIELD[0] + 1.0), ("goal_e", SPORTS_FIELD[2] - 2.0)):
        items.append(
            Structure(
                name,
                box_from_bounds(
                    (x_low, field_y - GOAL_HALF_WIDTH, 0.0), (x_low + 1.0, field_y + GOAL_HALF_WIDTH, GOAL_HEIGHT)
                ),
                "goal",
            )
        )
    return items


GOAL_HALF_WIDTH = 2.6
GOAL_HEIGHT = 2.1


STRUCTURES: Tuple[Structure, ...] = tuple(_structures())


def _solid_rects() -> List[Rect]:
    """Footprints that trees, lights, and signs must not overlap."""
    rects = [_rect_of(building.obstacle, 0.2) for building in BUILDINGS]
    rects.extend(_rect_of(building.canopy_obstacle, 0.3) for building in BUILDINGS)
    rects.extend(_rect_of(item.box, 0.5) for item in STRUCTURES)
    rects.extend([PARKING_LOT, SPORTS_FIELD])
    return rects


def _streets_sides():
    """Yield (axis, line, along_center, side) for every sidewalk segment."""
    for street_x in STREET_X:
        for block_y in BLOCK_Y:
            for side in (-1.0, 1.0):
                yield "y", street_x, block_y, side
    for street_y in STREET_Y:
        for block_x in BLOCK_X:
            for side in (-1.0, 1.0):
                yield "x", street_y, block_x, side
    for ring_x, inner in ((RING_X[0], 1.0), (RING_X[1], -1.0)):
        for block_y in BLOCK_Y:
            for side in (-1.0, 1.0):
                yield "y", ring_x, block_y, side
    for ring_y in RING_Y:
        for block_x in BLOCK_X:
            for side in (-1.0, 1.0):
                yield "x", ring_y, block_x, side


def _point(axis, line, along, offset):
    return (line + offset, along) if axis == "y" else (along, line + offset)


def _placements():
    solids = _solid_rects()
    trees: List[Tree] = []
    lights: List[StreetLight] = []
    signs: List[Sign] = []
    hedges: List[Hedge] = []

    def tree_fits(position, variant, scale):
        radius = TREE_VARIANTS[variant]["radius"] * scale
        rect = (position[0] - radius, position[1] - radius, position[0] + radius, position[1] + radius)
        if any(rects_overlap(rect, solid) for solid in solids):
            return False
        if not (FENCE_MIN[0] + 2.0 < position[0] < FENCE_MAX[0] - 2.0):
            return False
        if not (FENCE_MIN[1] + 2.0 < position[1] < FENCE_MAX[1] - 2.0):
            return False
        return all(
            math.dist(position, other.position_xy) >= radius + other.radius - 0.4
            for other in trees
        )

    def add_tree(position, shifts=(0.0,), axis=None):
        serial = len(trees) + 1
        variant = serial % len(TREE_VARIANTS)
        scale = 0.88 + 0.06 * (serial % 5)
        for shift in shifts:
            candidate = position
            if axis == "y":
                candidate = (position[0], position[1] + shift)
            elif axis == "x":
                candidate = (position[0] + shift, position[1])
            if tree_fits(candidate, variant, scale):
                trees.append(
                    Tree("tree_%03d" % serial, candidate, variant, scale, (serial * 1.7) % (2 * math.pi))
                )
                return

    for axis, line, along_center, side in _streets_sides():
        for along in (-9.0, 0.0, 9.0):
            add_tree(
                _point(axis, line, along_center + along, side * TREE_OFFSET),
                shifts=(0.0, -2.5, 2.5),
                axis=axis,
            )
        along = along_center + 4.5 * side
        position = _point(axis, line, along, side * LIGHT_OFFSET)
        facing = (-side, 0.0) if axis == "y" else (0.0, -side)
        light = StreetLight("light_%03d" % (len(lights) + 1), position, facing)
        if not any(rects_overlap(_rect_of(light.obstacle), solid) for solid in solids):
            lights.append(light)

        hedge_line = side * HEDGE_OFFSET
        pieces = [(along_center - 12.0, along_center - 2.0), (along_center + 2.0, along_center + 12.0)]
        for start, end in pieces:
            low = _point(axis, line, start, hedge_line - 0.3)
            high = _point(axis, line, end, hedge_line + 0.3)
            rect = (min(low[0], high[0]), min(low[1], high[1]), max(low[0], high[0]), max(low[1], high[1]))
            blocked = [solid for solid in solids if rects_overlap(rect, solid)]
            if blocked:
                continue
            hedges.append(Hedge("hedge_%03d" % (len(hedges) + 1), rect))

    for street_x in STREET_X:
        for street_y in STREET_Y:
            position = (street_x + 5.6, street_y - 5.6)
            sign = Sign("sign_%02d" % (len(signs) + 1), position, 0.0, len(signs) % 3)
            if not any(rects_overlap(_rect_of(sign.obstacle), solid) for solid in solids):
                signs.append(sign)
    for x_value in (GATE_X - 7.5, GATE_X + 9.0):
        signs.append(Sign("sign_%02d" % (len(signs) + 1), (x_value, -84.0), 0.0, len(signs) % 3))

    lot_x0, lot_y0, lot_x1, lot_y1 = PARKING_LOT
    for x_value in (lot_x0 + 2.0, (lot_x0 + lot_x1) / 2.0, lot_x1 - 2.5):
        lights.append(StreetLight("light_%03d" % (len(lights) + 1), (x_value, lot_y1 - 0.8), (0.0, -1.0)))

    # Park in the west belt and lawns around the service buildings.
    for index, x_value in enumerate(range(-90, -67, 6)):
        for y_value in range(-88, 91, 7):
            jitter = ((index * 7 + y_value * 3) % 5 - 2) * 0.6
            add_tree((float(x_value) + jitter, float(y_value) - jitter))
    for y_value in range(-88, 91, 9):
        add_tree((109.0, float(y_value)))
        add_tree((132.0, float(y_value)))
    for x_value in range(-88, 131, 10):
        add_tree((float(x_value), 92.0))
    for x_value in range(30, 46, 6):
        for y_value in (-88.0, -76.0):
            add_tree((float(x_value), y_value))
    return tuple(trees), tuple(lights), tuple(signs), tuple(hedges)


TREES, LIGHTS, SIGNS, HEDGES = _placements()


def _cars() -> Tuple[Car, ...]:
    lot_x0, lot_y0, lot_x1, lot_y1 = PARKING_LOT
    bays = []
    for row_y in (lot_y1 - 3.3, lot_y0 + 3.3):
        x_value = lot_x0 + 2.6
        while x_value < lot_x1 - 2.0:
            bays.append((x_value + 1.3, row_y))
            x_value += 2.6
    picked = [bays[index] for index in (1, 2, 5, 9, 10, 14, 19, 24, 27, 30, 31, 36) if index < len(bays)]
    return tuple(
        Car("car_%02d" % (index + 1), position, math.pi / 2.0 if index % 2 else -math.pi / 2.0, index % 6)
        for index, position in enumerate(picked)
    )


CARS: Tuple[Car, ...] = _cars()
PARKING_BAY_PITCH = 2.6


def obstacles() -> List[Tuple[str, BoxObstacle]]:
    """Named boxes for the planner, in the order written to YAML."""
    named = []
    for building in BUILDINGS:
        named.append((building.name, building.obstacle))
        penthouse = building.penthouse_obstacle
        if penthouse is not None:
            named.append((building.name + "_roof", penthouse))
        named.append((building.name + "_canopy", building.canopy_obstacle))
    for structure in STRUCTURES:
        named.append((structure.name, structure.box))
    for group in (TREES, LIGHTS, SIGNS, HEDGES, CARS):
        for item in group:
            named.append((item.name, item.obstacle))
    return named


def flat_obstacles() -> List[float]:
    values: List[float] = []
    for _, box in obstacles():
        values.extend(box.center)
        values.extend(box.size)
    return values


def flat_inspection_points() -> List[float]:
    return [value for point in INSPECTION_POINTS for value in point]


def road_rects() -> Dict[str, List[Rect]]:
    """Asphalt rectangles. North-south and east-west roads sit at different heights."""
    half = ROAD_HALF_WIDTH
    ring_low_y, ring_high_y = RING_Y[0] - half, RING_Y[1] + half
    ring_low_x, ring_high_x = RING_X[0] - half, RING_X[1] + half
    north_south = [(x - half, ring_low_y, x + half, ring_high_y) for x in STREET_X + RING_X]
    north_south.append((GATE_X - half, PUBLIC_ROAD_Y - half, GATE_X + half, ring_low_y))
    east_west = [(ring_low_x, y - half, ring_high_x, y + half) for y in STREET_Y + RING_Y]
    east_west.append((-160.0, PUBLIC_ROAD_Y - half, 200.0, PUBLIC_ROAD_Y + half))
    return {"north_south": north_south, "east_west": east_west}


def pavement_rects() -> List[Rect]:
    rects = [
        (bx - PAVEMENT_HALF, by - PAVEMENT_HALF, bx + PAVEMENT_HALF, by + PAVEMENT_HALF)
        for bx in BLOCK_X
        for by in BLOCK_Y
    ]
    half = ROAD_HALF_WIDTH
    ring_x0, ring_x1 = RING_X[0] - half, RING_X[1] + half
    ring_y0, ring_y1 = RING_Y[0] - half, RING_Y[1] + half
    width = SIDEWALK_OUTER - ROAD_HALF_WIDTH
    rects.extend(
        [
            (ring_x0 - width, ring_y0 - width, ring_x1 + width, ring_y0),
            (ring_x0 - width, ring_y1, ring_x1 + width, ring_y1 + width),
            (ring_x0 - width, ring_y0, ring_x0, ring_y1),
            (ring_x1, ring_y0, ring_x1 + width, ring_y1),
        ]
    )
    rects.append((GATE_X + half, FENCE_MIN[1] + 1.0, GATE_X + half + 3.0, ring_y0 - width))
    for building in BUILDINGS[-4:]:
        rects.append(_rect_of(building.obstacle, 3.0))
    return rects


def flat_rects() -> List[Rect]:
    """Areas where the terrain stays at z = 0 so paving does not intersect it."""
    rects = list(road_rects()["north_south"]) + list(road_rects()["east_west"])
    rects.extend(pavement_rects())
    rects.extend([PARKING_LOT, SPORTS_FIELD])
    rects.extend(_rect_of(building.obstacle, 1.0) for building in BUILDINGS)
    rects.extend(_rect_of(item.box, 1.5) for item in STRUCTURES)
    rects.append((-4.0, -4.0, 4.0, 4.0))
    return rects

