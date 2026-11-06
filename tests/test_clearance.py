"""Body size and clearance checks."""
import unittest
import numpy as np
from track_a.clearance import ROBOT_RADIUS, PLANNING_RADIUS, clearance_budget
from track_a.navigation import Navigator
from track_a.world import MARKER_XY


class ClearanceChecks(unittest.TestCase):
    def test_disc_contains_chassis_corners_and_wheel_extents(self):
        for x in [-.20, .08]:
            for y in [-.14, .14]:
                self.assertLessEqual(np.hypot(x, y), ROBOT_RADIUS+1e-12)
        self.assertGreaterEqual(ROBOT_RADIUS, np.hypot(.033, .153))

    def test_only_geometry_uncertainty_and_grid_contribute(self):
        small = clearance_budget(np.diag([.01**2, .02**2, .1]), .04)
        self.assertAlmostEqual(small['uncertainty_m'], .01)
        self.assertAlmostEqual(small['total_m'], PLANNING_RADIUS+.01+.04/np.sqrt(2))
        large = clearance_budget(np.diag([.20**2, .01**2, .1]), .04)
        self.assertAlmostEqual(large['uncertainty_m'], .10)
        self.assertGreater(large['total_m'], small['total_m'])

    def test_missing_route_waits_for_replanning_without_marker_search(self):
        driver = Navigator()
        xy = np.array([-1.8, .5])
        delta = MARKER_XY-xy
        pose = np.r_[xy, np.arctan2(delta[1], delta[0])]
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        v, _ = driver.command(pose, np.full(180, 5.), angles, float('inf'))
        self.assertEqual(v, 0)
        self.assertEqual(driver.command(pose, np.full(180, .1), angles, float('inf')), (0., 0.))
        self.assertEqual(driver.command(pose, np.full(180, 5.), angles, 0.), (0., 0.))


class CompactFootprintChecks(unittest.TestCase):
    def test_straight_gap_is_allowed_but_turning_corners_are_checked(self):
        from track_a.resilience import local_motion_clear
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        # 36cm gap: straight travel fits; a sideways turn does not.
        ranges = np.minimum(5., .18/np.maximum(np.abs(np.sin(angles)), 1e-8))
        self.assertTrue(local_motion_clear(.10, 0., ranges, angles))
        self.assertFalse(local_motion_clear(0., 1.5, ranges, angles))
        self.assertGreater(PLANNING_RADIUS, .153)
        self.assertLess(PLANNING_RADIUS, ROBOT_RADIUS)

    def test_guard_still_rejects_gap_smaller_than_actual_robot(self):
        from track_a.resilience import local_motion_clear
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        ranges = np.minimum(5., .14/np.maximum(np.abs(np.sin(angles)), 1e-8))
        self.assertFalse(local_motion_clear(.1, 0., ranges, angles))


if __name__ == '__main__':
    unittest.main()
