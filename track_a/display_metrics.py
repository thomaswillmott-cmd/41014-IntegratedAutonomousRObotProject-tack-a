"""Display-only diagnostics; kept separate from control."""
import mujoco
import numpy as np

from .world import WHEEL_RADIUS, WHEEL_SEPARATION


def rolling_slip(wheel_speed, forward_speed, yaw_rate):
    """Wheel surface minus wheel-hub forward speed, in m/s. Positive means overspeed; this measures rolling slip, not sideways skid."""
    hub_speed = np.array([forward_speed-yaw_rate*WHEEL_SEPARATION/2,
                          forward_speed+yaw_rate*WHEEL_SEPARATION/2])
    return np.asarray(wheel_speed)*WHEEL_RADIUS-hub_speed


def wheel_slip_for_evaluation(sim):
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY,
                            sim.model.body('base').id, velocity, 1)
    wheels = [float(sim.data.sensor('speed_left').data[0]),
              float(sim.data.sensor('speed_right').data[0])]
    return rolling_slip(wheels, velocity[3], velocity[2])
