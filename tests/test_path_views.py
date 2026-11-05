"""Verify reference switches and missing data cannot create artificial joins."""
import unittest
from track_a.path_views import path_runs


class PathTraceChecks(unittest.TestCase):
    def test_path_change_starts_a_separate_trace_without_zeroing_values(self):
        samples = [dict(time=0, path_id=1, tracking_error_m=.02),
                   dict(time=1, path_id=1, tracking_error_m=.03),
                   dict(time=2, path_id=2, tracking_error_m=.15),
                   dict(time=3, path_id=2, tracking_error_m=.10)]
        runs = path_runs(samples)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]['time'], [0, 1])
        self.assertEqual(runs[1]['time'], [2, 3])
        self.assertEqual(runs[1]['value'], [.15, .10])

    def test_missing_route_error_leaves_a_gap(self):
        samples = [dict(time=0, path_id=1, tracking_error_m=.02),
                   dict(time=1, path_id=1, tracking_error_m=None),
                   dict(time=2, path_id=1, tracking_error_m=.03)]
        self.assertEqual(len(path_runs(samples)), 2)


if __name__ == '__main__':
    unittest.main()
