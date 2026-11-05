"""Clean mode must bypass injections without bypassing sensor-based control."""
import unittest
from unittest.mock import Mock, patch
import numpy as np
from track_a.scenario import Scenario
from track_a.simulation import Simulation


class CleanModeChecks(unittest.TestCase):
    def test_clean_encoders_and_lidar_are_raw_without_rng_calls(self):
        sim = Simulation(camera=False, scenario=Scenario(clean=True))
        try:
            sim.rng = Mock()
            sim.rng.normal.side_effect = AssertionError('Injected noise in clean mode')
            previous = sim.previous_encoder.copy()
            with patch.object(sim.ekf, 'predict', wraps=sim.ekf.predict) as predict:
                sim.step()
            np.testing.assert_array_equal(predict.call_args.args[0], sim.encoders()-previous)
            np.testing.assert_array_equal(sim.ranges, sim.lidar.scan(sim.model, sim.data))
            sim.rng.normal.assert_not_called()
        finally:
            sim.close()

    def test_original_sensor_noise_remains_available(self):
        sim = Simulation(camera=False, scenario=Scenario(clean=False))
        try:
            with patch.object(sim, 'rng') as rng:
                rng.normal.side_effect = lambda mean, sigma, size: np.zeros(size)
                sim.step()
                self.assertEqual(rng.normal.call_count, 2)
        finally:
            sim.close()

    def test_clean_rejects_stress_injections(self):
        for option in ['noisy', 'dropout', 'disturbance']:
            with self.assertRaises(ValueError):
                Scenario(clean=True, **{option: True})
