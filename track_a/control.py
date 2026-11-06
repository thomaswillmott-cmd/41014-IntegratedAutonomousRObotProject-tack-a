"""Original proportional waypoint baseline; the LQR tracker is in navigation.py."""

import numpy as np

from .estimation import wrap
from .world import WAYPOINTS, WHEEL_RADIUS, WHEEL_SEPARATION


def wheel_commands(v, omega):
    wheels = np.array([v - omega*WHEEL_SEPARATION/2, v + omega*WHEEL_SEPARATION/2]) / WHEEL_RADIUS
    return wheels * min(1.0, 7.0 / max(7.0, float(np.max(np.abs(wheels)))))


class WaypointDriver:
    def __init__(self):
        self.index = 0
        self.finished = False
        self.status = 'Starting route'

    def command(self, estimate, ranges, angles):
        if self.finished:
            return 0.0, 0.0
        target = WAYPOINTS[self.index]
        delta = target - estimate[:2]
        distance = np.linalg.norm(delta)
        if distance < (0.055 if self.index == len(WAYPOINTS)-1 else 0.10):
            if self.index < len(WAYPOINTS)-1:
                self.index += 1
                return 0.0, 0.0
            error = wrap(-estimate[2])
            if abs(error) < 0.04:
                self.finished = True
                self.status = 'Dock approach complete (baseline)'
                return 0.0, 0.0
            self.status = 'Aligning at dock'
            return 0.0, float(np.clip(2*error, -0.6, 0.6))
        error = wrap(np.arctan2(delta[1], delta[0]) - estimate[2])
        omega = float(np.clip(2.2*error, -1, 1))
        v = min(0.18, 0.8*distance) * max(0, np.cos(error))**3
        if abs(error) > 0.65:
            v = 0.0
        front = ranges[np.abs(angles) < np.deg2rad(25)].min()
        # Local stop/slowdown only.
        v *= float(np.clip((front - 0.27)/0.35, 0, 1))
        self.status = f'Waypoint {self.index+1}/{len(WAYPOINTS)}'
        if front < 0.27 and v == 0 and abs(error) < 0.65:
            self.status = 'Obstacle stop: manual intervention or reset'
        return float(v), omega
