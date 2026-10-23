import importlib.util
import math
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import yaml

from park_inspection import campus
from park_inspection.planning import GridPlanner3D

PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORLD = PACKAGE_DIR / "worlds" / "park.sdf"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_campus", PACKAGE_DIR / "tools" / "generate_campus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _floats(text):
    return [float(value) for value in text.split()]


def _models():
    return {model.get("name"): model for model in ElementTree.parse(WORLD).getroot().iter("model")}


def _collision_bounds(model, collision):
    """World axis-aligned bounds of a primitive collision in a yaw-only model frame."""
    x0, y0, z0, _, _, yaw = _floats(model.find("pose").text)
    cx, cy, cz = _floats(collision.find("pose").text)[:3]
    center = (
        x0 + math.cos(yaw) * cx - math.sin(yaw) * cy,
        y0 + math.sin(yaw) * cx + math.cos(yaw) * cy,
        z0 + cz,
    )
    geometry = collision.find("geometry")
    if geometry.find("box") is not None:
        sx, sy, sz = _floats(geometry.find("box/size").text)
        if abs(math.sin(yaw)) > 0.5:
            sx, sy = sy, sx
        half = (sx / 2.0, sy / 2.0, sz / 2.0)
    elif geometry.find("cylinder") is not None:
        radius = float(geometry.find("cylinder/radius").text)
        half = (radius, radius, float(geometry.find("cylinder/length").text) / 2.0)
    elif geometry.find("sphere") is not None:
        radius = float(geometry.find("sphere/radius").text)
        half = (radius, radius, radius)
    else:
        return None
    return center, half


def _inside(center, half, box, tolerance=1e-3):
    return all(
        abs(center[axis] - box.center[axis]) + half[axis] <= box.size[axis] / 2.0 + tolerance
        for axis in range(3)
    )


def _overlap(first, second):
    return all(
        abs(first.center[axis] - second.center[axis]) < (first.size[axis] + second.size[axis]) / 2.0
        for axis in range(3)
    )


class TestGeneratedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = _generator()

    def test_world_yaml_and_meshes_match_the_layout(self):
        self.assertEqual(WORLD.read_text(encoding="utf-8"), self.generator.world_sdf(), "run tools/generate_campus.py")
        obstacles = (PACKAGE_DIR / "config" / "campus_obstacles.yaml").read_text(encoding="utf-8")
        self.assertEqual(obstacles, self.generator.obstacles_yaml(), "run tools/generate_campus.py")
        mesh_dir = PACKAGE_DIR / "worlds" / "meshes"
        expected = self.generator.mesh_files()
        self.assertEqual(sorted(path.name for path in mesh_dir.glob("*.obj")), sorted(expected))
        for name, text in expected.items():
            self.assertEqual((mesh_dir / name).read_text(encoding="utf-8"), text, name)
        for name in self.generator.textures():
            self.assertTrue((mesh_dir / name).is_file(), name)

    def test_every_mesh_and_texture_reference_resolves(self):
        text = WORLD.read_text(encoding="utf-8")
        root = ElementTree.fromstring(text)
        references = [uri.text for uri in root.iter("uri")] + [item.text for item in root.iter("albedo_map")]
        self.assertTrue(references)
        for reference in references:
            self.assertTrue((WORLD.parent / reference).is_file(), reference)

    def test_yaml_parameters_round_trip(self):
        data = yaml.safe_load(
            (PACKAGE_DIR / "config" / "campus_obstacles.yaml").read_text(encoding="utf-8")
        )["/**"]["ros__parameters"]
        for loaded, expected in zip(data["obstacles"], campus.flat_obstacles()):
            self.assertAlmostEqual(loaded, expected, places=2)
        self.assertEqual(len(data["obstacles"]), len(campus.flat_obstacles()))
        for key in ("obstacles", "inspection_points", "bounds_min", "bounds_max"):
            self.assertTrue(
                all(type(value) is float for value in data[key]),
                "%s must load as a ROS double array" % key,
            )
        self.assertIs(type(data["return_altitude"]), float)
        self.assertEqual(data["inspection_points"], campus.flat_inspection_points())
        self.assertEqual(tuple(data["bounds_min"]), campus.BOUNDS_MIN)
        self.assertEqual(tuple(data["bounds_max"]), campus.BOUNDS_MAX)


class TestWorldPhysics(unittest.TestCase):
    def test_every_link_has_collisions_and_inertia(self):
        for name, model in _models().items():
            for link in model.iter("link"):
                self.assertIsNotNone(link.find("collision"), name)
                inertial = link.find("inertial")
                self.assertIsNotNone(inertial, name)
                self.assertGreater(float(inertial.find("mass").text), 0.0, name)
                for axis in ("ixx", "iyy", "izz"):
                    self.assertGreater(float(inertial.find("inertia/" + axis).text), 0.0, name)

    def test_gravity_sun_and_shadows(self):
        world = ElementTree.parse(WORLD).getroot().find("world")
        self.assertLess(_floats(world.find("gravity").text)[2], -9.0)
        self.assertEqual(world.find("scene/shadows").text, "true")
        sun = world.find("light")
        self.assertEqual(sun.get("type"), "directional")
        self.assertEqual(sun.find("cast_shadows").text, "true")

    def test_building_body_collision_is_the_planner_box(self):
        models = _models()
        for building in campus.BUILDINGS:
            model = models[building.name]
            body = next(item for item in model.iter("collision") if item.get("name") == "body")
            center, half = _collision_bounds(model, body)
            for axis in range(3):
                self.assertAlmostEqual(center[axis], building.obstacle.center[axis], places=3)
                self.assertAlmostEqual(2.0 * half[axis], building.obstacle.size[axis], places=3)

    def test_collisions_stay_inside_planner_boxes(self):
        boxes = [box for _, box in campus.obstacles()]
        checked = 0
        for name, model in _models().items():
            if name == "ground":
                continue
            for collision in model.iter("collision"):
                center, half = _collision_bounds(model, collision)
                self.assertTrue(
                    any(_inside(center, half, box) for box in boxes),
                    "%s/%s" % (name, collision.get("name")),
                )
                checked += 1
        self.assertGreater(checked, 900)

    def test_trees_have_trunk_and_canopy(self):
        names = {item.get("name") for item in _models()["street_trees"].iter("collision")}
        for tree in campus.TREES:
            self.assertIn(tree.name + "_trunk", names)
            self.assertIn(tree.name + "_canopy", names)


class TestLayout(unittest.TestCase):
    def test_facades_leave_flyable_gaps(self):
        for index, first in enumerate(campus.BUILDINGS):
            for second in campus.BUILDINGS[index + 1:]:
                gap = max(
                    abs(first.center_xy[axis] - second.center_xy[axis])
                    - (first.size_xy[axis] + second.size_xy[axis]) / 2.0
                    for axis in range(2)
                )
                self.assertGreaterEqual(gap, 8.0 - 1e-6, (first.name, second.name))

    def test_trees_stay_off_buildings_and_the_spawn_pad(self):
        for tree in campus.TREES:
            box = tree.obstacle
            for building in campus.BUILDINGS:
                gap = max(
                    abs(box.center[axis] - building.center_xy[axis])
                    - (box.size[axis] + building.size_xy[axis]) / 2.0
                    for axis in range(2)
                )
                self.assertGreaterEqual(gap, 0.15, (tree.name, building.name))
            self.assertGreater(math.hypot(*tree.position_xy), 6.0, tree.name)

    def test_street_furniture_and_cars_stay_off_buildings(self):
        solids = [building.obstacle for building in campus.BUILDINGS]
        solids.extend(building.canopy_obstacle for building in campus.BUILDINGS)
        solids.extend(item.box for item in campus.STRUCTURES)
        for group in (campus.LIGHTS, campus.SIGNS, campus.HEDGES, campus.CARS):
            for item in group:
                for solid in solids:
                    self.assertFalse(_overlap(item.obstacle, solid), item.name)
        cars = [car.obstacle for car in campus.CARS]
        for index, first in enumerate(cars):
            for second in cars[index + 1:]:
                self.assertFalse(_overlap(first, second))

    def test_everything_inside_the_map_and_the_fence_inside_the_bounds(self):
        for name, box in campus.obstacles():
            for axis in range(3):
                self.assertGreaterEqual(box.center[axis] - box.size[axis] / 2.0, campus.BOUNDS_MIN[axis], name)
                self.assertLessEqual(box.center[axis] + box.size[axis] / 2.0, campus.BOUNDS_MAX[axis], name)
        self.assertGreaterEqual(len(campus.CARS), 8)
        self.assertGreater(len(campus.LIGHTS), 40)

    def test_route_threads_between_buildings(self):
        planner = GridPlanner3D(
            bounds_min=campus.BOUNDS_MIN,
            bounds_max=campus.BOUNDS_MAX,
            obstacles=[box for _, box in campus.obstacles()],
            resolution=1.0,
            clearance=2.5,
        )
        home = (0.0, 0.0, campus.RETURN_ALTITUDE)
        route = [home, *campus.INSPECTION_POINTS, home]
        highest = 0.0
        for start, goal in zip(route, route[1:]):
            path = planner.plan(start, goal)
            highest = max(highest, max(point[2] for point in path))
            for first, second in zip(path, path[1:]):
                self.assertTrue(planner._segment_is_free(first, second))
        roofs = sorted(
            building.height
            for building in campus.BUILDINGS
            if campus.RING_X[0] < building.center_xy[0] < campus.RING_X[1]
            and campus.RING_Y[0] < building.center_xy[1] < campus.RING_Y[1]
        )
        self.assertLess(highest, roofs[len(roofs) // 2])
