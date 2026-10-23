import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from park_inspection import mission_node


class TestOnlineReplanning(unittest.TestCase):
    def test_in_progress_replan_times_out_and_marks_mission_failed(self):
        failures = []
        node = SimpleNamespace(
            online_replan_lock=threading.Lock(),
            online_replan_result=None,
            online_replan_in_progress=True,
            online_replan_started_at=time.monotonic() - 2.0,
            online_replan_timeout=1.0,
            online_replan_failures=0,
            online_replan_timeouts=0,
            online_replan_duration_last_s=None,
            online_replan_duration_total_s=0.0,
            online_replan_duration_max_s=0.0,
            _fail_mission=failures.append,
        )

        mission_node.InspectionMission._process_online_replan_result(node)

        self.assertFalse(node.online_replan_in_progress)
        self.assertIsNone(node.online_replan_started_at)
        self.assertEqual(node.online_replan_failures, 1)
        self.assertEqual(node.online_replan_timeouts, 1)
        self.assertGreaterEqual(node.online_replan_duration_last_s, 2.0)
        self.assertEqual(len(failures), 1)
        self.assertIn("exceeded 1.0 seconds", failures[0])

    def test_unexpected_planner_exception_is_returned_to_mission(self):
        node = SimpleNamespace(
            online_replan_lock=threading.Lock(),
            online_replan_result=None,
        )
        planner_error = AssertionError("unexpected planner defect")

        with patch.object(mission_node, "GridPlanner3D", side_effect=planner_error):
            mission_node.InspectionMission._compute_online_replan(
                node,
                generation=3,
                route_index=1,
                start=(0.0, 0.0, 0.0),
                goal=(1.0, 1.0, 1.0),
                blocked_cells=set(),
                obstacles=(),
                bounds_min=(0.0, 0.0, 0.0),
                bounds_max=(2.0, 2.0, 2.0),
                resolution=1.0,
                clearance=0.0,
            )

        generation, route_index, path, error, duration = node.online_replan_result
        self.assertEqual((generation, route_index), (3, 1))
        self.assertIsNone(path)
        self.assertIn("AssertionError", error)
        self.assertIn("unexpected planner defect", error)
        self.assertGreaterEqual(duration, 0.0)


if __name__ == "__main__":
    unittest.main()
