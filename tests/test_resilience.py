"""Fault recovery and wheel-feedback checks."""
import unittest
from unittest.mock import patch
import numpy as np
from track_a.resilience import (WheelFeedback, RecoverySupervisor, CameraDriveCalibration,
                                MarkerRelocalizer, camera_pose, local_motion_clear, marker_view_route)
from track_a.estimation import PoseEKF
from track_a.simulation import Simulation
from track_a.scenario import Scenario

class ResilienceChecks(unittest.TestCase):
    def setUp(self):
        self.angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        self.ranges = np.full(180, 5.)
        self.pose = np.array([0., 0., 0.])
        self.P = np.eye(3)*.0001

    def test_wheel_feedback_rejects_unknown_motor_gain_bias(self):
        loop = WheelFeedback()
        speed = np.zeros(2)
        rng = np.random.default_rng(3)
        gains = np.array([.94, 1.06])
        for _ in range(600):
            measured_delta = (speed+rng.normal(0, .1, 2))*.02
            output = loop.update([4., 4.], measured_delta)
            speed += .4*(output*gains-speed)
        self.assertLess(np.max(abs(speed-4.)), .08)
        self.assertGreater(np.max(abs(4*gains-4)), .2)
        for _ in range(500):
            self.assertLessEqual(np.max(abs(loop.update([100., -100.], [0., 0.]))), 7.8)
        np.testing.assert_array_equal(loop.update([0., 0.], [0., 0.]), [0., 0.])

    def test_camera_loss_continues_slower_without_timeout_or_marker_search(self):
        supervisor = RecoverySupervisor()
        supervisor.camera_result(0., True, self.pose)
        for t in [1.1, 21., 120.]:
            # Camera loss must not invent corrections or change the route direction.
            command = supervisor.command(t, self.pose, np.eye(3)*.1, t,
                                         (.2, .4), self.ranges, self.angles, 'NAVIGATE')
            np.testing.assert_allclose(command, (.15, .3))
            self.assertEqual(supervisor.state, 'SLOW')
            self.assertIsNone(supervisor.view_path)
            self.assertIsNone(supervisor.retreat_target)
        supervisor.camera_result(120.2, True, self.pose)
        supervisor.camera_result(120.4, True, self.pose)
        supervisor.camera_result(120.6, True, self.pose)
        self.assertEqual(supervisor.state, 'NORMAL')
        self.assertFalse(supervisor.recovered)  # Keep the learned map.

    def test_missing_route_during_camera_loss_stops_without_searching(self):
        supervisor = RecoverySupervisor()
        command = supervisor.command(10., self.pose, self.P, 10., (.2, .4),
                                     self.ranges, self.angles, 'NAVIGATE', path_available=False)
        self.assertEqual(command, (0., 0.))
        self.assertEqual(supervisor.state, 'SLOW')
        self.assertEqual(supervisor.view_goal, None)

    def test_consistent_observations_required_to_resume(self):
        supervisor = RecoverySupervisor()
        supervisor.encoder_fault(1.)
        supervisor.camera_result(2., True, self.pose)
        supervisor.camera_result(2.2, True, self.pose)
        self.assertEqual(supervisor.state, 'RELOCALISE')
        supervisor.camera_result(2.4, True, self.pose)
        self.assertEqual(supervisor.state, 'NORMAL')
        self.assertTrue(supervisor.recovered)

    def test_local_clearance_blocks_obstacle_and_allows_open_space(self):
        self.assertTrue(local_motion_clear(.04, .4, self.ranges, self.angles))
        self.assertFalse(local_motion_clear(.04, .4, np.full(180, .1), self.angles))
        bad = self.ranges.copy(); bad[20] = np.nan
        self.assertFalse(local_motion_clear(.04, .4, bad, self.angles))

    def test_marker_inverse_and_multiframe_relocalisation(self):
        pose = np.array([1.8, 1.2, .1])
        measurement = PoseEKF.observation(pose)
        np.testing.assert_allclose(camera_pose(measurement, pose[2]), pose, atol=1e-12)
        recovery = MarkerRelocalizer()
        for i in range(4):
            self.assertIsNone(recovery.observe(i*.2, measurement, pose[2], [0., 0.]))
        result = recovery.observe(.8, measurement, pose[2], [0., 0.])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[0], pose)
        self.assertTrue(np.all(np.linalg.eigvalsh(result[1]) > 0))

    def test_marker_motion_or_inconsistency_prevents_reset(self):
        recovery = MarkerRelocalizer()
        for i in range(10):
            m = np.array([1.+.3*(i%2), 0.])
            self.assertIsNone(recovery.observe(i*.2, m, 0., [0., 0.]))
        self.assertIsNone(recovery.observe(2., [1., 0.], 0., [1., 1.]))

    def test_camera_drive_calibration_is_slow_bounded_and_requires_motion(self):
        calibration = CameraDriveCalibration()
        for i in range(15):
            distance = .03*i
            calibration.observe(.2*i, [1.+distance*1.04, 1., 0.], np.full(2, distance/.033), 1.)
        self.assertGreaterEqual(calibration.updates, 1)
        self.assertTrue(np.all(calibration.scale > 1.))
        self.assertTrue(np.all(calibration.scale < 1.01))
        before = calibration.scale.copy()
        for i in range(15, 30):
            calibration.observe(.2*i, [1.+.42*1.04, 1., 0.], np.full(2, .42/.033), 1.)
        np.testing.assert_array_equal(before, calibration.scale)

    def test_marker_viewpoint_route_respects_inflated_occupancy(self):
        from track_a.planning import OccupancyMap
        grid = OccupancyMap()
        grid.inflate(.35)
        pose = np.array([-1., -1., 0.])
        path = marker_view_route(grid, pose, [])
        self.assertIsNotNone(path)
        self.assertTrue(all(grid.clear_segment(a, b) for a, b in zip(path[:-1], path[1:])))
        self.assertGreater(np.linalg.norm(path[-1]-pose[:2]), .35)

    def test_large_pose_inconsistency_relocalises_from_images_without_truth(self):
        sim = Simulation(seed=8, scenario=Scenario(seed=8, clean=True, start=np.array([1.5, 1.4, 0.])))
        try:
            sim.ekf.x += np.array([.45, -.2, .2])
            with patch.object(sim, 'record_evaluation'), patch.object(sim, 'true_pose_for_evaluation',
                    side_effect=AssertionError('Truth leaked into recovery')):
                for _ in range(220):
                    sim.step()
            self.assertTrue(any('Camera relocalisation' in e['event'] for e in sim.events))
            self.assertTrue(any(e.get('state') == 'NORMAL' for e in sim.recovery.events))
            self.assertLess(np.linalg.norm(sim.ekf.x[:2]-sim.true_pose_for_evaluation()[:2]), .07)
            self.assertEqual(sim.collision_steps, 0)
        finally:
            sim.close()

    def test_marker_search_waits_for_alignment_before_retracing(self):
        supervisor = RecoverySupervisor()
        supervisor.encoder_fault(0.)
        supervisor.breadcrumbs.append(np.array([-.3, 0., 0.]))
        pose = np.array([0., 0., -2.5])
        command = supervisor.command(3., pose, self.P, 10., (0., 0.), self.ranges, self.angles, 'NAVIGATE')
        self.assertEqual(command[0], 0.)
        self.assertIsNone(supervisor.retreat_target)
        pose[2] = np.arctan2(1.4, 2.82)
        supervisor.command(4., pose, self.P, 10., (0., 0.), self.ranges, self.angles, 'NAVIGATE')
        supervisor.command(5.6, pose, self.P, 10., (0., 0.), self.ranges, self.angles, 'NAVIGATE')
        self.assertIsNotNone(supervisor.retreat_target)
        self.assertEqual(supervisor.command(61., pose, self.P, 100., (0., 0.), self.ranges, self.angles, 'NAVIGATE'), (0., 0.))

    def test_actual_camera_outage_slows_without_freezing_planner_and_resumes(self):
        sim = Simulation(seed=9, scenario=Scenario(seed=9, clean=True, start=np.array([1., 1.4, 0.])))
        capture = sim.camera.capture
        def temporary_loss(data):
            image, measurement = capture(data)
            return image, None if .6 < data.time < 3. else measurement
        try:
            with patch.object(sim.camera, 'capture', side_effect=temporary_loss):
                while sim.data.time < 5.:
                    sim.step()
                    if 1.5 < sim.data.time < 2.5 and sim.recovery.state == 'SLOW':
                        self.assertFalse(sim.driver.freeze_plan)
            states = [e.get('state') for e in sim.recovery.events]
            self.assertNotIn('PAUSE', states)
            self.assertNotIn('RELOCALISE', states)
            self.assertIn('SLOW', states)
            self.assertEqual(sim.recovery.state, 'NORMAL')
            self.assertEqual(sim.collision_steps, 0)
        finally:
            sim.close()

    def test_camera_recovery_allows_replanning_before_route_fault_retrigger(self):
        supervisor = RecoverySupervisor()
        supervisor.encoder_fault(0.)
        for t in [1., 1.2, 1.4]:
            supervisor.camera_result(t, True, self.pose)
        self.assertEqual(supervisor.state, 'NORMAL')
        supervisor.command(1.4, self.pose, self.P, 0., (0., 0.), self.ranges, self.angles,
                           'NAVIGATE', path_available=False)
        self.assertEqual(supervisor.state, 'NORMAL')

    def test_recovery_budget_accounts_for_route_and_stays_bounded(self):
        supervisor = RecoverySupervisor()
        supervisor.encoder_fault(10.)
        supervisor.budget_route(20., np.zeros(3), np.array([[0., 0.], [-1.5, 0.]]))
        self.assertGreater(supervisor.motion_until, 55.)
        supervisor.budget_route(100., np.zeros(3), np.array([[0., 0.], [100., 0.]]))
        self.assertLessEqual(supervisor.motion_until, 110.)
        self.assertLessEqual(supervisor.stop_until, 125.)

    def test_off_axis_camera_absence_continues_slowly_without_search(self):
        supervisor = RecoverySupervisor()
        pose = np.array([0., 0., np.arctan2(1.4, 2.82)-np.deg2rad(25)])
        supervisor.camera_result(0., True, pose)
        supervisor.command(1.1, pose, self.P, 1.1, (.1, 0.), self.ranges, self.angles, 'NAVIGATE')
        self.assertEqual(supervisor.state, 'SLOW')

    def test_saved_route_survives_failed_global_replan_only_with_camera_loss_policy(self):
        from track_a.navigation import Navigator
        driver = Navigator()
        pose = np.array([-2., -1.5, 0.])
        driver.observe(pose, self.P, self.ranges, self.angles, 0.)
        self.assertGreaterEqual(len(driver.path), 2)
        previous = driver.path.copy()
        driver.keep_path_without_camera = True
        with patch('track_a.navigation.astar', return_value=None), patch.object(driver.map, 'clear_segment', return_value=False):
            driver.observe(pose, self.P, self.ranges, self.angles, 3.)
            np.testing.assert_array_equal(driver.path, previous)
            driver.keep_path_without_camera = False
            driver.observe(pose, self.P, self.ranges, self.angles, 6.)
            self.assertEqual(len(driver.path), 0)

    def test_saved_route_never_overrides_live_lidar_obstacle_stop(self):
        supervisor = RecoverySupervisor()
        command = supervisor.command(2., self.pose, self.P, 2., (.2, 0.), np.full(180, .1),
                                     self.angles, 'NAVIGATE')
        self.assertEqual(command, (0., 0.))
        self.assertEqual(supervisor.state, 'SLOW')
