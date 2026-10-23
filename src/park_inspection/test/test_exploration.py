import math
import unittest

from park_inspection.exploration import (
    FrontierInformationStructure,
    plan_exploration,
)
from park_inspection.occupancy import OccupancyMap3D
from park_inspection.planning import GridPlanner3D


def observed_corridor(y_values, max_range=6.0):
    grid = OccupancyMap3D(
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(20.0, 16.0, 6.0),
        resolution=1.0,
        clearance=0.0,
    )
    for y_value in y_values:
        grid.integrate_scan(
            (0.5, y_value, 2.5),
            [(30.0, y_value, 2.5)],
            max_range=max_range,
        )
    return grid


def paint_bundle(grid, y_origin, max_range):
    updated = []
    for y_index in range(3):
        for z_index in range(3):
            grid.integrate_scan(
                (0.5, y_origin + y_index, 1.5 + z_index),
                [(40.0, y_origin + y_index, 1.5 + z_index)],
                max_range=max_range,
            )
            updated.extend(grid.last_updated_cells)
    return updated


class TestFrontierStructure(unittest.TestCase):
    def test_free_cells_touching_unknown_space_form_one_cluster(self):
        grid = observed_corridor((2.5, 3.5, 4.5))
        structure = FrontierInformationStructure(viewpoint_radius=3.0)
        changed = structure.update(grid, grid.last_updated_cells)

        self.assertTrue(changed)
        self.assertEqual(len(structure.clusters), 1)
        self.assertGreaterEqual(len(structure.clusters[0].cells), 3)
        self.assertTrue(structure.clusters[0].viewpoints)
        self.assertGreater(structure.clusters[0].viewpoints[0].coverage, 0)

    def test_a_later_scan_moves_the_frontier_and_keeps_a_distant_cluster(self):
        grid = OccupancyMap3D(
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(20.0, 20.0, 8.0),
            resolution=1.0,
            clearance=0.0,
        )
        updated = paint_bundle(grid, 2.0, 6.0)
        updated.extend(paint_bundle(grid, 14.0, 6.0))
        structure = FrontierInformationStructure(viewpoint_radius=3.0, min_cluster_cells=4)
        structure.update(grid, updated)
        distant_cells = {
            cell
            for cluster in structure.clusters
            if cluster.average[1] > 10.0
            for cell in cluster.cells
        }
        near_before = max(
            cell[0]
            for cluster in structure.clusters
            if cluster.average[1] < 8.0
            for cell in cluster.cells
        )

        moved = paint_bundle(grid, 2.0, 10.0)
        structure.update(grid, moved)
        kept = {
            cell
            for cluster in structure.clusters
            if cluster.average[1] > 10.0
            for cell in cluster.cells
        }
        near_after = max(
            cell[0]
            for cluster in structure.clusters
            if cluster.average[1] < 8.0
            for cell in cluster.cells
        )
        self.assertEqual(kept, distant_cells)
        self.assertGreater(near_after, near_before)

    def test_a_long_frontier_is_split_along_its_long_axis(self):
        grid = observed_corridor(tuple(0.5 + index for index in range(10)))
        structure = FrontierInformationStructure(
            max_cluster_cells=3, viewpoint_radius=3.0
        )
        structure.update(grid, grid.last_updated_cells)

        self.assertGreaterEqual(len(structure.clusters), 3)
        for cluster in structure.clusters:
            self.assertLessEqual(len(cluster.cells), 3)


class TestExplorationPlan(unittest.TestCase):
    def test_tour_visits_each_cluster_and_the_path_reaches_the_first(self):
        grid = observed_corridor((2.5, 3.5))
        grid.integrate_scan((0.5, 12.5, 2.5), [(30.0, 12.5, 2.5)], max_range=6.0)
        grid.integrate_scan((0.5, 13.5, 2.5), [(30.0, 13.5, 2.5)], max_range=6.0)
        structure = FrontierInformationStructure(viewpoint_radius=3.0)
        structure.update(grid, [(0, 0, 0), (19, 15, 5)])
        planner = GridPlanner3D(
            bounds_min=grid.bounds_min,
            bounds_max=grid.bounds_max,
            obstacles=[],
            resolution=1.0,
            clearance=0.0,
            is_blocked=grid.blocks,
        )
        plan = plan_exploration((1.5, 3.5, 2.5), 0.0, structure.clusters, planner)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.tour), len(structure.clusters))
        self.assertLess(plan.tour[0].position[1], plan.tour[-1].position[1])
        self.assertGreater(len(plan.path), 1)
        self.assertEqual(plan.path[0], (1.5, 3.5, 2.5))
        self.assertLess(math.dist(plan.path[-1], plan.local_viewpoints[0].position), 1.5)

    def test_no_frontier_does_not_invent_a_route(self):
        grid = OccupancyMap3D((0.0, 0.0, 0.0), (8.0, 8.0, 4.0), resolution=1.0)
        planner = GridPlanner3D(
            bounds_min=grid.bounds_min,
            bounds_max=grid.bounds_max,
            obstacles=[],
            resolution=1.0,
            clearance=0.0,
        )
        self.assertIsNone(plan_exploration((1.5, 1.5, 1.5), 0.0, [], planner))
