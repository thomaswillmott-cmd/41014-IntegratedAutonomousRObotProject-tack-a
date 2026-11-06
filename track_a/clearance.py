"""Planner clearance from body size and estimated uncertainty."""
import numpy as np

# Body bounds from world.py: x=[-.20,.08], y=[-.14,.14]; wheel y=+/-.144, halfwidth .009.
# Full turning radius includes the corners.
CHASSIS_RADIUS = float(np.hypot(.20, .14))
WHEEL_RADIUS_BOUND = float(np.hypot(.033, .144+.009))
ROBOT_RADIUS = max(CHASSIS_RADIUS, WHEEL_RADIUS_BOUND)


# Compact planning radius; turning corners still need the live body check.
PLANNING_RADIUS = .16
BODY_REAR = .20
BODY_FRONT = .08
BODY_HALF_WIDTH = .153
LOCAL_NOISE_MARGIN = .015


def clearance_budget(covariance, resolution, sigma_factor=.5):
    """Clearance terms in metres. Half a sigma is a tuning allowance, not a 95% safety bound; uncertainty stays uncapped."""
    xy = np.asarray(covariance, dtype=float)[:2, :2]
    sigma = float(np.sqrt(max(0., np.linalg.eigvalsh((xy+xy.T)/2).max())))
    grid = float(resolution/np.sqrt(2))
    return dict(physical_radius_m=ROBOT_RADIUS, planning_radius_m=PLANNING_RADIUS, uncertainty_m=sigma_factor*sigma,
                grid_allowance_m=grid, total_m=PLANNING_RADIUS+sigma_factor*sigma+grid,
                sigma_factor=sigma_factor)
