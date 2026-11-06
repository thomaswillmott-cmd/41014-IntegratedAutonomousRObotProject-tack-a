"""Physical motion, camera and EKF checks."""

import unittest

import mujoco
import numpy as np

from track_a.control import WaypointDriver, wheel_commands
from track_a.estimation import PoseEKF
from track_a.sensing import Lidar, MarkerCamera
from track_a.world import START, load_world


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.model = load_world()
        self.data = mujoco.MjData(self.model)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)

    def test_stationary_robot_and_wall_ranges(self):
        self.assertLess(np.linalg.norm(self.data.qvel[:3]), 0.01)
        ranges = Lidar(self.model).scan(self.model, self.data)
        self.assertTrue(np.all(np.isfinite(ranges)))
        self.assertTrue(np.all(ranges > 0))
        self.assertAlmostEqual(ranges.min(), 0.95, delta=0.08)

    def test_encoder_prediction_matches_physical_forward_and_turn(self):
        ekf = PoseEKF(START)
        previous = self.data.qpos[7:9].copy()
        for command in [(0.15, 0)]*100 + [(0, 0.6)]*100:
            self.data.ctrl[:] = wheel_commands(*command)
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            current = self.data.qpos[7:9].copy()
            ekf.predict(current-previous)
            previous = current
        rotation = self.data.body('base').xmat.reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.assertLess(np.linalg.norm(ekf.x[:2]-self.data.qpos[:2]), 0.05)
        self.assertLess(abs(ekf.x[2]-yaw), 0.10)

    def test_rendered_marker_measurement_and_ekf_update(self):
        self.data.qpos[:3] = [1.5, 1.4, 0]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        camera = MarkerCamera(self.model)
        try:
            image, measurement = camera.capture(self.data)
            self.assertEqual(image.shape, (480, 640, 3))
            self.assertIsNotNone(measurement)
            self.assertAlmostEqual(measurement[0], 1.21, delta=0.04)
            self.assertAlmostEqual(measurement[1], 0, delta=0.03)
            ekf = PoseEKF([1.52, 1.42, 0.01])
            self.assertTrue(ekf.update(measurement))
            self.assertTrue(np.all(np.linalg.eigvalsh(ekf.P) > 0))
            self.data.qpos[:2] = [-2, -1.5]
            mujoco.mj_forward(self.model, self.data)
            _, absent = camera.capture(self.data)
            self.assertIsNone(absent)
        finally:
            camera.close()

    def test_camera_outlier_is_rejected(self):
        ekf = PoseEKF(START)
        before = ekf.x.copy()
        self.assertFalse(ekf.update([0.2, 2.5]))
        np.testing.assert_array_equal(ekf.x, before)

    def test_front_obstacle_stops_forward_command(self):
        driver = WaypointDriver()
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        v, _ = driver.command(START.copy(), np.full(180, 0.2), angles)
        self.assertEqual(v, 0)


if __name__ == '__main__':
    unittest.main()
