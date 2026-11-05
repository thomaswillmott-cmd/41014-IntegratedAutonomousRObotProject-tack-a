"""Grid refinement must retain world coverage and physical inflation distances."""
import unittest
import numpy as np
from track_a.planning import OccupancyMap, RESOLUTION, ORIGIN, MAP_MAX
from track_a.clearance import clearance_budget


class GridResolutionChecks(unittest.TestCase):
    def test_finer_grid_retains_world_bounds_and_rounding_error_bound(self):
        grid = OccupancyMap()
        self.assertEqual(RESOLUTION, .02)
        self.assertEqual(grid.logodds.shape, (257, 305))
        np.testing.assert_allclose(grid.xy((0, 0)), ORIGIN)
        np.testing.assert_allclose(grid.xy((304, 256)), MAP_MAX)
        rng = np.random.default_rng(8)
        for point in rng.uniform(ORIGIN, MAP_MAX, (100, 2)):
            self.assertLessEqual(np.linalg.norm(grid.xy(grid.cell(point))-point), RESOLUTION/np.sqrt(2)+1e-12)
        self.assertTrue(np.all(grid.known[[0, -1], :]))
        self.assertTrue(np.all(grid.known[:, [0, -1]]))

    def test_refinement_reduces_only_grid_component_of_clearance(self):
        covariance = np.diag([.03**2, .04**2, .02**2])
        old, new = clearance_budget(covariance, .04), clearance_budget(covariance, RESOLUTION)
        self.assertEqual(old['physical_radius_m'], new['physical_radius_m'])
        self.assertEqual(old['uncertainty_m'], new['uncertainty_m'])
        self.assertAlmostEqual(new['grid_allowance_m'], .02/np.sqrt(2))
        self.assertAlmostEqual(old['total_m']-new['total_m'], .02/np.sqrt(2))
