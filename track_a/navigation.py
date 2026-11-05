"""Hybrid navigation: LiDAR map/A*, LQR tracking, camera-gated pose docking.

Inputs are estimated pose/covariance, scans and camera age. No truth API.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import solve_discrete_are

from .estimation import wrap
from .planning import OccupancyMap, astar, RESOLUTION
from .scenario import GOAL, STAGING
from .clearance import clearance_budget
from .world import MARKER_XY


@lru_cache(maxsize=32)
def lqr_gain(speed=.18, effort=1.):
    dt = .02
    A = np.eye(3)
    A[1, 2] = speed*dt
    B = np.array([[dt, 0], [0, 0], [0, dt]])
    Q = np.diag([6., 14., 3.])
    R = np.diag([2., 1.])*effort
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R+B.T@P@B, B.T@P@A)
    return K, A, B, Q, R


class Navigator:
    def __init__(self, objective='time', controller='lqr', known_obstacles=None):
        self.map = OccupancyMap(known_obstacles)
        self.objective = objective
        self.controller = controller
        self.path = np.empty((0, 2))
        self.index = 0
        self.phase = 'NAVIGATE'
        self.finished = False
        self.status = 'Scanning and planning'
        self.replans = 0
        self.last_plan = -10.
        self.tracking_error = 0.
        self.hold = 0.
        self.obstacle_stop = False
        self.freeze_plan = False
        self.keep_path_without_camera = False
        self.tracking_reference = None
        self.clearance = clearance_budget(np.zeros((3, 3)), RESOLUTION)

    def observe(self, estimate, covariance, ranges, angles, now):
        self.map.update(estimate, angles, ranges)
        self.clearance = clearance_budget(covariance, RESOLUTION)
        self.map.inflate(self.clearance['total_m'])
        if self.phase != 'NAVIGATE' or self.freeze_plan:
            return
        blocked = len(self.path) < 2
        if not blocked:
            target = self.path[min(self.index+1, len(self.path)-1)]
            blocked = not self.map.clear_segment(estimate[:2], target)
        if blocked or now-self.last_plan > 2.:
            if now-self.last_plan < .5:
                return
            candidate = astar(self.map, estimate[:2], STAGING, estimate[2], self.objective)
            self.last_plan = now
            self.replans += 1
            if candidate is not None:
                self.path = candidate
                self.index = 0
            elif blocked and self.keep_path_without_camera and len(self.path) >= 2:
                # Keep the last reference during visual loss. It is not permission
                # to traverse a new obstacle: the supervisor checks live relative
                # LiDAR against the physical swept body before each movement.
                self.status = 'Following saved route with live LiDAR checks'
            elif blocked:
                self.path = np.empty((0, 2))
                self.status = 'No safe route - stopped, waiting for new scans'

    def command(self, estimate, ranges, angles, camera_age=float('inf')):
        self.obstacle_stop = False
        self.tracking_reference = None
        if self.finished:
            return 0., 0.
        if self.phase == 'NAVIGATE':
            if np.linalg.norm(estimate[:2]-STAGING) < .11:
                self.phase = 'ACQUIRE'
            elif len(self.path) < 2:
                # A missing route is a planning problem. Wait for the next scan
                # and A* attempt rather than turning/reversing to seek a marker.
                self.status = 'No safe route - stopped, LiDAR replanning active'
                return 0., 0.
            else:
                while self.index < len(self.path)-2 and np.linalg.norm(estimate[:2]-self.path[self.index+1]) < .11:
                    self.index += 1
                a, b = self.path[self.index:self.index+2]
                self.tracking_reference = (a.copy(), b.copy())
                vector = b-a
                length = np.linalg.norm(vector)
                if length < 1e-6:
                    return 0., 0.
                tangent = vector/length
                heading = np.arctan2(tangent[1], tangent[0])
                heading_error = wrap(estimate[2]-heading)
                self.status = f'{self.controller.upper()} tracking | plans {self.replans}'
                if abs(heading_error) > .55:
                    return 0., float(np.clip(-2*heading_error, -.9, .9))
                projection = float(np.clip((estimate[:2]-a)@tangent, 0, length))
                reference = a + min(length, projection+.06)*tangent
                normal = np.array([-tangent[1], tangent[0]])
                error = np.array([(estimate[:2]-reference)@tangent,
                                  (estimate[:2]-reference)@normal, heading_error])
                self.tracking_error = float(abs(error[1]))
                distance = np.linalg.norm(b-estimate[:2])
                speed = float(np.clip(.8*distance, .06, .18))
                if self.controller == 'lqr':
                    K, *_ = lqr_gain(round(speed/0.02)*0.02)
                    correction = -K@error
                    v, omega = speed+correction[0], correction[1]
                else:
                    desired = np.arctan2(b[1]-estimate[1], b[0]-estimate[0])
                    v, omega = speed, 2.2*wrap(desired-estimate[2])
                v = float(np.clip(v, 0, .20))
                omega = float(np.clip(omega, -1., 1.))
                return self.safe(v, omega, ranges, angles)
        if self.phase == 'ACQUIRE':
            self.status = 'Aligning camera with dock marker'
            error = wrap(-estimate[2])
            if abs(error) > .06:
                return 0., float(np.clip(2*error, -.7, .7))
            if camera_age < .5:
                self.phase = 'DOCK'
            else:
                self.status = 'Waiting for marker - stopped'
                return 0., 0.
        if self.phase in ('DOCK', 'VERIFY'):
            if camera_age > .8:
                self.status = 'Marker lost - safe docking stop'
                self.hold = 0.
                return 0., 0.
            delta = GOAL[:2]-estimate[:2]
            distance = np.linalg.norm(delta)
            if distance < .04 or (self.phase == 'VERIFY' and distance < .07):
                self.phase = 'VERIFY'
                v = 0.
                omega = float(np.clip(2*wrap(-estimate[2]), -.4, .4))
                if abs(wrap(estimate[2])) < .08:
                    self.hold += .02
                    if self.hold > .6:
                        self.phase = 'DOCKED'
                        self.finished = True
                        self.status = 'DOCKED | camera-confirmed estimated pose'
                        return 0., 0.
                self.status = 'Dock pose verification'
                return v, omega
            self.phase = 'DOCK'
            self.hold = 0.
            bearing = np.arctan2(delta[1], delta[0])
            alpha = wrap(bearing-estimate[2])
            direction = 1.
            if abs(alpha) > np.pi/2:
                direction = -1.
                bearing = wrap(bearing+np.pi)
                alpha = wrap(bearing-estimate[2])
            beta = wrap(-bearing)
            v = direction*min(.10, .65*distance)*max(0., np.cos(alpha))
            omega = float(np.clip(2*alpha-.65*beta, -.65, .65))
            self.status = 'Visual docking | fresh camera + EKF pose'
            return self.safe(v, omega, ranges, angles)
        return 0., 0.

    def safe(self, v, omega, ranges, angles):
        front = ranges[np.abs(angles) < np.deg2rad(30)].min()
        if v > 0:
            factor = float(np.clip((front-.17)/.25, 0, 1))
            self.obstacle_stop = factor == 0
            v *= factor
        elif v < 0:
            rear = ranges[np.abs(angles) > np.deg2rad(150)].min()
            v *= float(np.clip((rear-.20)/.25, 0, 1))
        return float(v), float(omega)
