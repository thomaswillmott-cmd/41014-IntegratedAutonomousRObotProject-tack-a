"""EKF prediction from wheels, correction from camera. No simulator pose input."""

import numpy as np

from .world import CAMERA_OFFSET, MARKER_XY, WHEEL_RADIUS, WHEEL_SEPARATION


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PoseEKF:
    def __init__(self, initial_pose, q_scale=1.):
        self.x = np.array(initial_pose, dtype=float).copy()
        self.P = np.diag([0.02**2, 0.02**2, np.deg2rad(2)**2])
        self.updates = 0
        self.rejected = 0
        self.last_nis = float('nan')
        self.q_scale = q_scale

    def predict(self, wheel_delta):
        left, right = np.asarray(wheel_delta) * WHEEL_RADIUS
        ds = (left + right) / 2
        da = (right - left) / WHEEL_SEPARATION
        heading = self.x[2] + da / 2
        c, s = np.cos(heading), np.sin(heading)
        F = np.array([[1, 0, -ds*s], [0, 1, ds*c], [0, 0, 1]])
        G = np.array([[c/2 + ds*s/(2*WHEEL_SEPARATION), c/2 - ds*s/(2*WHEEL_SEPARATION)],
                      [s/2 - ds*c/(2*WHEEL_SEPARATION), s/2 + ds*c/(2*WHEEL_SEPARATION)],
                      [-1/WHEEL_SEPARATION, 1/WHEEL_SEPARATION]])
        # Assumed wheel-distance noise; still a tuning value.
        variance = (0.02 * np.abs([left, right]))**2 + 1e-9
        self.x += [ds*c, ds*s, da]
        self.x[2] = wrap(self.x[2])
        self.P = F @ self.P @ F.T + G @ np.diag(variance) @ G.T
        # Extra uncertainty for slip and model error beyond encoder noise.
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        slip = abs(ds)*np.diag([.0009, .0036, .0006]) + abs(da)*np.diag([1e-5, 1e-5, .0009])
        self.P += self.q_scale * rotation@slip@rotation.T

    @staticmethod
    def observation(x):
        camera = x[:2] + CAMERA_OFFSET * np.array([np.cos(x[2]), np.sin(x[2])])
        delta = MARKER_XY - camera
        return np.array([np.linalg.norm(delta), wrap(np.arctan2(delta[1], delta[0]) - x[2])])

    def update(self, measurement, yaw=None):
        expected = self.observation(self.x)
        H = np.empty((2, 3))
        for j in range(3):
            shifted = self.x.copy()
            shifted[j] += 1e-5
            difference = self.observation(shifted) - expected
            difference[1] = wrap(difference[1])
            H[:, j] = difference / 1e-5
        # Allow more range uncertainty at longer distances.
        R = np.diag([(0.015 + 0.015 * measurement[0]**2)**2, np.deg2rad(1.5)**2])
        measurement = np.asarray(measurement)
        if yaw is not None:
            H = np.vstack([H, [0, 0, 1]])
            expected = np.r_[expected, self.x[2]]
            measurement = np.r_[measurement, yaw]
            # Allow for weaker PnP heading at long range or frontal views.
            expanded = np.zeros((3, 3))
            expanded[:2, :2] = R
            expanded[2, 2] = np.deg2rad(5 + 2*measurement[0])**2
            R = expanded
        innovation = np.asarray(measurement) - expected
        innovation[1] = wrap(innovation[1])
        if yaw is not None:
            innovation[2] = wrap(innovation[2])
        S = H @ self.P @ H.T + R
        self.last_nis = float(innovation @ np.linalg.solve(S, innovation))
        if self.last_nis > (11.34 if yaw is not None else 9.21):
            self.rejected += 1
            return False
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ innovation
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        self.updates += 1
        return True
