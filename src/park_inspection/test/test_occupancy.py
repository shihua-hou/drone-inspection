import math
import unittest

from park_inspection.occupancy import OccupancyMap3D, compare_inspection_plans
from park_inspection.planning import BoxObstacle, GridPlanner3D
from park_inspection.sensing import (
    cloud_to_map_rays,
    iter_cloud_xyz,
    iter_laser_scan_points,
)


def small_map(**overrides):
    options = {
        "bounds_min": (0.0, 0.0, 0.0),
        "bounds_max": (10.0, 6.0, 6.0),
        "resolution": 1.0,
        "clearance": 0.0,
    }
    options.update(overrides)
    return OccupancyMap3D(**options)


class _Field:
    def __init__(self, name, offset):
        self.name = name
        self.offset = offset
        self.datatype = 7


class _Cloud:
    def __init__(self, points):
        import struct

        self.fields = [_Field("x", 0), _Field("y", 4), _Field("z", 8)]
        self.point_step = 12
        self.width = len(points)
        self.height = 1
        self.is_bigendian = False
        self.data = b"".join(struct.pack("<fff", *point) for point in points)


class TestOccupancyMap3D(unittest.TestCase):
    def test_ray_marks_endpoint_occupied_and_traversed_cells_free(self):
        grid = small_map()
        accepted = grid.integrate_scan((0.5, 2.5, 2.5), [(4.2, 2.5, 2.5)], max_range=15.0)

        self.assertEqual(accepted, 1)
        self.assertIn((4, 2, 2), grid.occupied_cells())
        self.assertNotIn((1, 2, 2), grid.occupied_cells())
        self.assertLess(grid._log_odds[(1, 2, 2)], 0.0)

    def test_repeated_hits_clamp_and_misses_can_clear_a_cell(self):
        grid = small_map()
        for _ in range(20):
            grid.integrate_scan((0.5, 2.5, 2.5), [(4.2, 2.5, 2.5)], max_range=15.0)
        self.assertAlmostEqual(grid._log_odds[(4, 2, 2)], grid._clamp_max)

        for _ in range(20):
            grid.integrate_scan((0.5, 2.5, 2.5), [(8.5, 2.5, 2.5)], max_range=15.0)
        self.assertNotIn((4, 2, 2), grid.occupied_cells())

    def test_ground_and_max_range_returns_do_not_create_occupied_cells(self):
        grid = small_map()
        grid.integrate_scan(
            (1.5, 1.5, 2.0),
            [(1.5, 1.5, 0.1), (8.5, 1.5, 2.0)],
            max_range=4.0,
            ground_z=0.4,
        )
        self.assertEqual(grid.occupied_cells(), [])
        self.assertGreater(len(grid._log_odds), 0)

    def test_clearance_blocks_neighbors_of_an_occupied_cell(self):
        grid = small_map(clearance=1.0)
        grid.integrate_scan((0.5, 2.5, 2.5), [(4.2, 2.5, 2.5)], max_range=15.0)

        self.assertTrue(grid.blocks((4.5, 2.5, 2.5)))
        self.assertTrue(grid.blocks((5.5, 2.5, 2.5)))
        self.assertFalse(grid.blocks((8.5, 2.5, 2.5)))

    def test_blocked_generation_changes_only_when_blocked_voxels_change(self):
        grid = small_map()
        self.assertEqual(grid.blocked_generation, 0)

        for _ in range(2):
            grid.integrate_scan(
                (0.5, 2.5, 2.5), [(4.2, 2.5, 2.5)], max_range=15.0
            )

        self.assertEqual(grid.blocked_generation, 1)

    def test_agreement_counts_a_scanned_box_and_ignores_unseen_boxes(self):
        obstacle = BoxObstacle(center=(4.5, 2.5, 2.5), size=(1.0, 1.0, 1.0))
        hidden = BoxObstacle(center=(8.5, 2.5, 2.5), size=(1.0, 1.0, 1.0))
        grid = small_map()
        grid.integrate_scan((0.5, 2.5, 2.5), [(4.5, 2.5, 2.5)], max_range=15.0)
        stats = grid.agreement([obstacle, hidden])

        self.assertGreaterEqual(stats["true_positive_cells"], 1)
        self.assertEqual(stats["precision"], 1.0)
        self.assertEqual(stats["recall_observed"], 1.0)
        self.assertGreater(stats["prior_occupied_cells"], stats["true_positive_cells"])

    def test_straight_ray_visits_each_cell_once_in_order(self):
        grid = small_map()
        cells = grid.cells_along((0.5, 1.5, 1.5), (3.5, 1.5, 1.5))

        self.assertEqual(cells, [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)])


class TestPlanComparison(unittest.TestCase):
    def test_sensor_only_route_cuts_through_an_unseen_wall(self):
        bounds_min = (0.0, -2.0, 0.0)
        bounds_max = (8.0, 4.0, 4.0)
        wall = BoxObstacle(center=(4.0, 1.5, 1.5), size=(1.0, 8.0, 2.0))
        grid = OccupancyMap3D(
            bounds_min, bounds_max, resolution=1.0, clearance=0.0
        )
        waypoints = [(1.0, 1.5, 1.5), (6.5, 1.5, 1.5)]

        comparison = compare_inspection_plans(
            bounds_min, bounds_max, [wall], 1.0, 0.0, grid, waypoints
        )

        self.assertTrue(comparison["prior_plan_ok"])
        self.assertTrue(comparison["online_plan_ok"])
        self.assertTrue(comparison["fused_plan_ok"])
        self.assertGreater(
            comparison["prior_path_length_m"], comparison["online_path_length_m"]
        )
        self.assertEqual(
            comparison["fused_path_length_m"], comparison["prior_path_length_m"]
        )
        self.assertGreater(comparison["online_path_prior_intersections"], 0)

    def test_fused_route_avoids_a_sensed_obstacle_absent_from_the_prior(self):
        bounds_min = (0.0, 0.0, 0.0)
        bounds_max = (8.0, 6.0, 4.0)
        grid = OccupancyMap3D(
            bounds_min, bounds_max, resolution=1.0, clearance=0.0
        )
        grid.integrate_scan((1.5, 2.5, 1.5), [(3.6, 2.5, 1.5)], max_range=10.0)
        waypoints = [(1.0, 2.5, 1.5), (6.5, 2.5, 1.5)]

        comparison = compare_inspection_plans(
            bounds_min, bounds_max, [], 1.0, 0.0, grid, waypoints
        )

        self.assertTrue(comparison["fused_plan_ok"])
        self.assertGreater(
            comparison["fused_path_length_m"], comparison["prior_path_length_m"]
        )

    def test_planner_rejects_a_goal_blocked_only_by_occupancy(self):
        grid = small_map()
        grid.integrate_scan((0.5, 2.5, 2.5), [(4.2, 2.5, 2.5)], max_range=15.0)
        planner = GridPlanner3D(
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(10.0, 6.0, 6.0),
            obstacles=[],
            resolution=1.0,
            clearance=0.0,
            is_blocked=grid.blocks,
        )

        with self.assertRaisesRegex(ValueError, "Goal position"):
            planner.plan((0.5, 2.5, 2.5), (4.5, 2.5, 2.5))


class TestSensing(unittest.TestCase):
    def test_x_forward_cloud_uses_the_body_mount_and_yaw(self):
        yaw = math.pi / 2.0
        quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        origin, endpoints = cloud_to_map_rays(
            [(5.0, 0.0, 0.0)],
            vehicle_position=(10.0, 20.0, 3.0),
            quaternion=quaternion,
            sensor_position_body=(1.0, 0.0, 0.0),
            convention="x_forward",
        )

        self.assertAlmostEqual(origin[0], 10.0, places=6)
        self.assertAlmostEqual(origin[1], 21.0, places=6)
        self.assertAlmostEqual(origin[2], 3.0, places=6)
        self.assertAlmostEqual(endpoints[0][0], 10.0, places=6)
        self.assertAlmostEqual(endpoints[0][1], 26.0, places=6)

    def test_optical_forward_point_matches_x_forward(self):
        identity = (0.0, 0.0, 0.0, 1.0)
        _, optical = cloud_to_map_rays(
            [(0.0, 0.0, 4.0)],
            (0.0, 0.0, 1.0),
            identity,
            sensor_position_body=(0.0, 0.0, 0.0),
            convention="optical",
        )
        _, forward = cloud_to_map_rays(
            [(4.0, 0.0, 0.0)],
            (0.0, 0.0, 1.0),
            identity,
            sensor_position_body=(0.0, 0.0, 0.0),
            convention="x_forward",
        )

        self.assertTrue(all(math.isclose(a, b) for a, b in zip(optical[0], forward[0])))

    def test_point_cloud_parser_skips_non_finite_points_and_applies_stride(self):
        cloud = _Cloud([(1.0, 2.0, 3.0), (math.nan, 0.0, 0.0), (4.0, 5.0, 6.0)])

        self.assertEqual(
            list(iter_cloud_xyz(cloud, stride=1)),
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        self.assertEqual(list(iter_cloud_xyz(cloud, stride=3)), [(1.0, 2.0, 3.0)])


class _Scan:
    def __init__(self):
        self.angle_min = -math.pi / 2.0
        self.angle_increment = math.pi / 2.0
        self.range_min = 0.2
        self.range_max = 10.0
        self.ranges = [4.0, math.inf, 2.0]


class TestLaserScan(unittest.TestCase):
    def test_scan_places_hits_and_keeps_max_range_for_dropouts(self):
        points = list(iter_laser_scan_points(_Scan(), stride=1))

        self.assertAlmostEqual(points[0][0], 0.0, places=6)
        self.assertAlmostEqual(points[0][1], -4.0, places=6)
        self.assertAlmostEqual(points[1][0], 10.0, places=6)
        self.assertAlmostEqual(points[1][1], 0.0, places=6)
        self.assertAlmostEqual(points[2][0], 0.0, places=6)
        self.assertAlmostEqual(points[2][1], 2.0, places=6)

    def test_lidar_return_marks_the_cell_at_sensor_height(self):
        grid = small_map()
        scan = _Scan()
        scan.angle_min = 0.0
        scan.angle_increment = 0.1
        scan.ranges = [4.0]
        origin, endpoints = cloud_to_map_rays(
            iter_laser_scan_points(scan),
            vehicle_position=(0.5, 2.5, 2.2),
            quaternion=(0.0, 0.0, 0.0, 1.0),
            sensor_position_body=(0.0, 0.0, 0.3),
            convention="x_forward",
        )
        grid.integrate_scan(origin, endpoints, max_range=15.0, ground_z=0.4)

        self.assertIn((4, 2, 2), grid.occupied_cells())


if __name__ == "__main__":
    unittest.main()
