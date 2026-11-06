"""Error measurements and scoring across route changes."""
import unittest
import numpy as np
from track_a.evaluation import PathEvaluation, nearest_on_path


class PositionErrorChecks(unittest.TestCase):
    def test_distance_uses_finite_segments_and_corners(self):
        path = [[0, 0], [2, 0], [2, 2]]
        point, distance = nearest_on_path([1, .3], path)
        np.testing.assert_allclose(point, [1, 0])
        self.assertAlmostEqual(distance, .3)
        point, distance = nearest_on_path([-1, 0], path)
        np.testing.assert_allclose(point, [0, 0])
        self.assertAlmostEqual(distance, 1.)
        _, distance = nearest_on_path([2.4, 1], path)
        self.assertAlmostEqual(distance, .4)

    def test_replan_does_not_zero_truth_error_or_clear_accumulated_score(self):
        tracker = PathEvaluation()
        estimate = np.array([1., 0., 0.])
        tracker.activate([[0, 0], [2, 0]], 'nav', 0, estimate, [1, .2, 0], 'initial')
        tracker.sample(0, estimate, [1, .2, 0], 'NAVIGATE')
        tracker.activate([[1, 0], [3, 0]], 'nav', 1, estimate, [1, .5, 0], 'replan')
        tracker.sample(1, estimate, [1, .5, 0], 'NAVIGATE')
        self.assertEqual(len(tracker.history), 2)
        self.assertEqual(len(tracker.samples), 2)
        self.assertAlmostEqual(tracker.samples[-1]['estimated_tracking_error_m'], 0)
        self.assertAlmostEqual(tracker.samples[-1]['tracking_error_m'], .5)
        self.assertAlmostEqual(tracker.tracking_rmse, np.sqrt((.2**2+.5**2)/2))

    def test_unchanged_reference_does_not_make_fake_versions(self):
        tracker = PathEvaluation()
        for t in range(3):
            tracker.activate([[0, 0], [2, 0]], 'nav', t, [0, 0, 0], [0, 0, 0], 'refresh')
        self.assertEqual(len(tracker.history), 1)

    def test_no_path_is_missing_data_and_idle_does_not_dilute_score(self):
        tracker = PathEvaluation()
        tracker.sample(0, [0, 0, 0], [0, .3, 0], 'NO_PATH')
        self.assertIsNone(tracker.tracking_rmse)
        tracker.activate([[0, 0], [2, 0]], 'nav', 1, [0, 0, 0], [0, .3, 0], 'initial')
        tracker.sample(1, [0, 0, 0], [0, .3, 0], 'NAVIGATE')
        tracker.sample(2, [0, 0, 0], [0, 0, 0], 'DOCKED', scored=False)
        self.assertAlmostEqual(tracker.tracking_rmse, .3)


class AveragePositionChecks(unittest.TestCase):
    def test_average_is_mean_distance_and_excludes_unscored_samples(self):
        evaluation = PathEvaluation()
        self.assertIsNone(evaluation.average_position_error)
        evaluation.sample(0, np.array([.03, .04, 0]), np.zeros(3), 'NAV')
        evaluation.sample(.1, np.array([0, .01, 0]), np.zeros(3), 'NAV')
        evaluation.sample(.2, np.array([10, 0, 0]), np.zeros(3), 'MANUAL', False)
        self.assertAlmostEqual(evaluation.average_position_error, .03)
        self.assertAlmostEqual(evaluation.current['average_position_error_m'], .03)
        self.assertGreater(evaluation.estimation_rmse, evaluation.average_position_error)
        self.assertIsNone(PathEvaluation().average_position_error)


if __name__ == '__main__':
    unittest.main()
