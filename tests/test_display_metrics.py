"""Camera display, speed error and slip checks."""
import unittest
from unittest.mock import Mock
import numpy as np
from track_a.display_metrics import rolling_slip
from track_a.world import WHEEL_RADIUS, WHEEL_SEPARATION
from track_a.simulation import Simulation
from track_a.scenario import Scenario
from track_a.dashboard import dashboard


class DisplayMetricChecks(unittest.TestCase):
    def test_pure_rolling_forward_and_turning_have_zero_slip(self):
        for v, w in [(.2, 0.), (0., .6), (.1, -.4)]:
            wheel_speed = np.array([v-w*WHEEL_SEPARATION/2, v+w*WHEEL_SEPARATION/2])/WHEEL_RADIUS
            np.testing.assert_allclose(rolling_slip(wheel_speed, v, w), [0., 0.], atol=1e-12)
        np.testing.assert_allclose(rolling_slip([2., 0.], 0., 0.), [.066, 0.])

    def test_dropped_frame_is_black_then_live_image_returns(self):
        sim = Simulation(seed=2, scenario=Scenario(seed=2, dropout=True, start=np.array([1., 1.4, 0.])))
        try:
            generator = sim.rng
            sim.rng = Mock(wraps=generator)
            sim.rng.random.return_value = 0.
            sim.step()
            self.assertEqual(sim.display_camera_state, 'DROPPED FRAME')
            self.assertIsNone(sim.measurement)
            self.assertFalse(np.any(sim.image))
            panel = dashboard(sim)
            self.assertFalse(np.any(panel[100:160, :640]))
            sim.rng.random.return_value = 1.
            for _ in range(10):
                sim.step()
            self.assertEqual(sim.display_camera_state, 'LIVE')
            self.assertTrue(np.any(sim.image))
            # Missing marker does not mean camera blackout.
            sim.measurement = None
            panel = dashboard(sim)
            self.assertTrue(np.any(panel[100:160, :640]))
        finally:
            sim.close()
