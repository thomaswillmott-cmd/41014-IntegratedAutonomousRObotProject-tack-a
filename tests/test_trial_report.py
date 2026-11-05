import unittest
import numpy as np

from track_a.trial_report import initial_plan, reference_deviations, rms


class ReferenceReportChecks(unittest.TestCase):
    def test_first_usable_plan_is_retained_after_replanning(self):
        history = [dict(kind='navigation', points=[], time=0, path_id=1),
                   dict(kind='navigation', points=[[0, 0], [2, 0]], time=1, path_id=2),
                   dict(kind='navigation', points=[[0, 1], [2, 1]], time=2, path_id=3)]
        version = initial_plan(history)
        self.assertEqual(version['path_id'], 2)
        sample = dict(scored='True', time='3', phase='NAVIGATE', true_x='1', true_y='1')
        self.assertAlmostEqual(reference_deviations([sample], version)[0], 1.)
        self.assertEqual(version['points'], [[0, 0], [2, 0]])

    def test_missing_reference_and_docking_are_not_zero_error(self):
        version = dict(points=[[0, 0], [2, 0]], time=1)
        sample = dict(scored='True', time='3', phase='DOCK', true_x='3', true_y='1')
        self.assertTrue(np.isnan(reference_deviations([sample], version)[0]))
        sample['phase'] = 'NAVIGATE'
        sample['time'] = '0'
        self.assertTrue(np.isnan(reference_deviations([sample], version)[0]))
        self.assertTrue(np.isnan(reference_deviations([sample], None)[0]))
        self.assertIsNone(rms([np.nan]))


if __name__ == '__main__':
    unittest.main()
