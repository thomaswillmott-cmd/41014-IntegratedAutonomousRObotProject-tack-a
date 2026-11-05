"""Footprint-based planner clearance, without simulator truth input."""
import numpy as np

# Matches the runtime collision geometry in world.py. Axle-origin chassis box
# spans x=[-.20,.08], y=[-.14,.14]; wheel centres y=+/-.144, halfwidth .009.
# A disc encloses the robot for all yaw angles, including turn-in-place.
CHASSIS_RADIUS = float(np.hypot(.20, .14))
WHEEL_RADIUS_BOUND = float(np.hypot(.033, .144+.009))
ROBOT_RADIUS = max(CHASSIS_RADIUS, WHEEL_RADIUS_BOUND)


# Compact translation planning envelope. This does NOT enclose turning corners;
# every automatic Navigator command also needs a full-body local swept check.
PLANNING_RADIUS = .16
BODY_REAR = .20
BODY_FRONT = .08
BODY_HALF_WIDTH = .153
LOCAL_NOISE_MARGIN = .015


def clearance_budget(covariance, resolution, sigma_factor=.5):
    """Return physical, uncertainty and grid contributions in metres.

    Half a major-axis sigma is an engineering allowance, NOT a 95% collision-risk
    bound. No clipping hides high estimator uncertainty. Ground-truth error is
    excluded because it is unavailable to the controller on a real robot.
    """
    xy = np.asarray(covariance, dtype=float)[:2, :2]
    sigma = float(np.sqrt(max(0., np.linalg.eigvalsh((xy+xy.T)/2).max())))
    grid = float(resolution/np.sqrt(2))
    return dict(physical_radius_m=ROBOT_RADIUS, planning_radius_m=PLANNING_RADIUS, uncertainty_m=sigma_factor*sigma,
                grid_allowance_m=grid, total_m=PLANNING_RADIUS+sigma_factor*sigma+grid,
                sigma_factor=sigma_factor)
