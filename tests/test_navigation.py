"""Independent geometry, control and failure-mode regression checks."""
import unittest
from unittest.mock import patch
import numpy as np

from track_a.estimation import PoseEKF, wrap
from track_a.model import measurement_jacobian, controllability
from track_a.navigation import Navigator, lqr_gain
from track_a.planning import OccupancyMap, astar
from track_a.scenario import Scenario, STAGING
from track_a.simulation import Simulation
from track_a.world import START


class NavigationChecks(unittest.TestCase):
    def test_lqr_local_closed_loop_is_stable_and_controllable(self):
        for speed in [.06, .12, .18, .20]:
            K, A, B, _, _ = lqr_gain(speed)
            self.assertEqual(np.linalg.matrix_rank(controllability(A, B)), 3)
            self.assertLess(max(abs(np.linalg.eigvals(A-B@K))), 1)

    def test_analytic_sensor_jacobian_matches_finite_difference(self):
        x = np.array([.4, -.2, .6])
        H = measurement_jacobian(x, False)
        numeric = np.zeros_like(H)
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = 1e-6
            delta = PoseEKF.observation(x+dx)-PoseEKF.observation(x-dx)
            delta[1] = wrap(delta[1])
            numeric[:, j] = delta/2e-6
        np.testing.assert_allclose(H, numeric, atol=1e-6)

    def test_astar_detours_without_corner_cutting(self):
        grid = OccupancyMap([('block', 'box', (0., 0.), (.4, .8))])
        grid.inflate(.32)
        for objective in ['distance', 'time']:
            path = astar(grid, [-2, 0], [2, 0], objective=objective)
            self.assertIsNotNone(path)
            self.assertGreater(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)), 4)
            self.assertTrue(all(grid.clear_segment(a, b) for a, b in zip(path[:-1], path[1:])))

    def test_blocked_goal_returns_no_route(self):
        grid = OccupancyMap([('block', 'box', tuple(STAGING), (.4, .4))])
        grid.inflate(.32)
        self.assertIsNone(astar(grid, START[:2]))

    def test_docking_stops_when_camera_is_lost(self):
        navigator = Navigator()
        navigator.phase = 'DOCK'
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        self.assertEqual(navigator.command(np.array([2., 1.4, 0]), np.full(180, 5.), angles, 2.), (0., 0.))
        self.assertFalse(navigator.finished)

    def test_random_layout_reproduces_and_changes_with_seed(self):
        self.assertEqual(Scenario(seed=3, layout='random').obstacles(), Scenario(seed=3, layout='random').obstacles())
        self.assertNotEqual(Scenario(seed=3, layout='random').obstacles(), Scenario(seed=4, layout='random').obstacles())

    def test_control_keeps_running_without_truth_evaluation(self):
        sim = Simulation(camera=False)
        try:
            # Evaluation is disabled, and reading truth raises. Sensing/navigation
            # must still work. This tests the dependency boundary in execution.
            with patch.object(sim, 'record_evaluation'), patch.object(sim, 'true_pose_for_evaluation', side_effect=AssertionError('Truth leaked')):
                for _ in range(20):
                    sim.step()
            self.assertTrue(np.all(np.isfinite(sim.ekf.x)))
        finally:
            sim.close()

    def test_physics_contact_monitor_detects_robot_obstacle_overlap(self):
        sim = Simulation(camera=False)
        try:
            sim.data.qpos[:3] = [-.25, .35, 0]
            import mujoco
            mujoco.mj_forward(sim.model, sim.data)
            sim.monitor_contacts()
            self.assertGreater(sim.collision_steps, 0)
        finally:
            sim.close()


if __name__ == '__main__':
    unittest.main()
