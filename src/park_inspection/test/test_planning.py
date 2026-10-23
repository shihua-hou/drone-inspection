import math
import unittest

from park_inspection.planning import (
    BoxObstacle,
    GridPlanner3D,
    flatten_cell_index,
    grid_shape_for_bounds,
    path_intersects_grid_cells,
    point_in_grid_cells,
    unflatten_cell_index,
)


def planner():
    return GridPlanner3D(
        bounds_min=(-2.0, -5.0, 0.0),
        bounds_max=(12.0, 5.0, 10.0),
        obstacles=[BoxObstacle(center=(5.0, 0.0, 4.0), size=(2.0, 6.0, 8.0))],
        resolution=0.5,
        clearance=0.5,
    )


class TestGridPlanner3D(unittest.TestCase):
    def test_compact_occupancy_indices_round_trip(self):
        shape = grid_shape_for_bounds(
            (-2.0, -5.0, 0.0), (12.0, 5.0, 10.0), 0.5
        )
        cell = (7, 12, 19)

        self.assertEqual(shape, (28, 20, 20))
        self.assertEqual(
            unflatten_cell_index(flatten_cell_index(cell, shape), shape), cell
        )

    def test_compact_occupancy_indices_reject_out_of_bounds_values(self):
        with self.assertRaisesRegex(ValueError, "outside the grid"):
            flatten_cell_index((3, 1, 3), (3, 2, 4))
        with self.assertRaisesRegex(ValueError, "outside the grid"):
            unflatten_cell_index(24, (3, 2, 4))

    def test_path_routes_around_3d_obstacle_with_clearance(self):
        grid = planner()
        path = grid.plan((0.0, 0.0, 2.0), (10.0, 0.0, 2.0))

        self.assertEqual(path[0], (0.0, 0.0, 2.0))
        self.assertEqual(path[-1], (10.0, 0.0, 2.0))
        self.assertGreater(len(path), 2)
        for start, end in zip(path, path[1:]):
            self.assertTrue(grid._segment_is_free(start, end))

    def test_path_can_use_altitude_to_clear_obstacle(self):
        grid = planner()
        path = grid.plan((0.0, 0.0, 2.0), (10.0, 0.0, 9.0))

        self.assertEqual(path[-1], (10.0, 0.0, 9.0))
        self.assertTrue(any(point[2] > 8.0 for point in path))

    def test_blocked_goal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Goal position"):
            planner().plan((0.0, 0.0, 2.0), (5.0, 0.0, 4.0))

    def test_fully_blocked_corridor_reports_no_route(self):
        grid = GridPlanner3D(
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(10.0, 2.0, 2.0),
            obstacles=[
                BoxObstacle(center=(5.0, 1.0, 1.0), size=(1.0, 10.0, 10.0))
            ],
            resolution=1.0,
            clearance=0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "No collision-free route"):
            grid.plan((0.0, 1.0, 1.0), (10.0, 1.0, 1.0))

    def test_signed_distance_is_positive_outside_and_negative_inside(self):
        obstacle = BoxObstacle(center=(1.0, 2.0, 3.0), size=(2.0, 4.0, 6.0))

        self.assertTrue(
            math.isclose(obstacle.signed_distance((3.0, 2.0, 3.0)), 1.0)
        )
        self.assertTrue(
            math.isclose(obstacle.signed_distance((1.0, 2.0, 3.0)), -1.0)
        )

    def test_online_grid_block_check_respects_bounds_and_voxel_indices(self):
        blocked = {(2, 1, 1)}

        self.assertTrue(
            point_in_grid_cells(
                (2.5, 1.5, 1.5),
                (0.0, 0.0, 0.0),
                (5.0, 4.0, 4.0),
                1.0,
                blocked,
            )
        )
        self.assertFalse(
            point_in_grid_cells(
                (1.5, 1.5, 1.5),
                (0.0, 0.0, 0.0),
                (5.0, 4.0, 4.0),
                1.0,
                blocked,
            )
        )
        self.assertTrue(
            point_in_grid_cells(
                (5.1, 1.5, 1.5),
                (0.0, 0.0, 0.0),
                (5.0, 4.0, 4.0),
                1.0,
                blocked,
            )
        )

    def test_online_grid_detects_obstacle_along_segment_between_waypoints(self):
        blocked = {(2, 1, 1)}
        bounds_min = (0.0, 0.0, 0.0)
        bounds_max = (5.0, 4.0, 4.0)

        self.assertTrue(
            path_intersects_grid_cells(
                (0.5, 1.5, 1.5),
                [(4.5, 1.5, 1.5)],
                bounds_min,
                bounds_max,
                1.0,
                blocked,
            )
        )
        self.assertFalse(
            path_intersects_grid_cells(
                (0.5, 0.5, 1.5),
                [(4.5, 0.5, 1.5)],
                bounds_min,
                bounds_max,
                1.0,
                blocked,
            )
        )


if __name__ == "__main__":
    unittest.main()
