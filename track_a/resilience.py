"""Sensor-only fault handling and drive feedback. No simulator state or fault flags."""
from collections import deque
import numpy as np

from .estimation import wrap
from .clearance import BODY_REAR, BODY_FRONT, BODY_HALF_WIDTH, LOCAL_NOISE_MARGIN
from .world import CAMERA_OFFSET, MARKER_XY, WHEEL_RADIUS, WHEEL_SEPARATION


def camera_pose(measurement, yaw):
    """Invert known-marker range/bearing plus image-derived heading."""
    distance, bearing = measurement
    camera = MARKER_XY-distance*np.array([np.cos(yaw+bearing), np.sin(yaw+bearing)])
    return np.r_[camera-CAMERA_OFFSET*np.array([np.cos(yaw), np.sin(yaw)]), wrap(yaw)]


def recovery_route_seconds(path, heading):
    vectors = np.diff(path, axis=0)
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    return (np.linalg.norm(vectors, axis=1).sum()/.06
            + np.abs(wrap(np.diff(np.r_[heading, headings]))).sum()/.45)


class MarkerRelocalizer:
    """Require five independent, stationary, consistent image poses before recovery."""
    def __init__(self):
        self.frames = deque(maxlen=5)

    def observe(self, now, measurement, yaw, wheel_speed):
        if yaw is None or max(abs(np.asarray(wheel_speed))) > .3:
            self.frames.clear()
            return None
        pose = camera_pose(measurement, yaw)
        if self.frames and now-self.frames[-1][0] > .81:
            self.frames.clear()
        self.frames.append((now, pose, float(measurement[0])))
        if len(self.frames) < 5:
            return None
        poses = np.array([f[1] for f in self.frames])
        mean = np.r_[np.mean(poses[:, :2], axis=0),
                     np.arctan2(np.sin(poses[:, 2]).mean(), np.cos(poses[:, 2]).mean())]
        if max(np.linalg.norm(poses[:, :2]-mean[:2], axis=1)) > .10 or max(abs(wrap(poses[:, 2]-mean[2]))) > .08:
            return None
        distance = float(np.mean([f[2] for f in self.frames]))
        # Conservative, correlated pose covariance propagated from image range,
        # bearing and heading errors. Includes an irreducible calibration floor.
        def invert(z):
            return camera_pose(z[:2], z[2])
        z = np.r_[measurement, yaw]
        columns = []
        for i in range(3):
            difference = invert(z+np.eye(3)[i]*1e-5)-invert(z)
            difference[2] = wrap(difference[2])
            columns.append(difference/1e-5)
        J = np.column_stack(columns)
        noise = np.diag([(.015+.015*distance**2)**2, np.deg2rad(1.5)**2,
                         np.deg2rad(5+2*distance)**2])
        covariance = J@noise@J.T/5 + np.diag([.015**2, .015**2, np.deg2rad(2)**2])
        self.frames.clear()
        return mean, covariance


class WheelFeedback:
    """Bounded PI around noisy angle-derived wheel speed, with acceleration limiting."""
    def __init__(self):
        self.speed = np.zeros(2)
        self.target = np.zeros(2)
        self.integral = np.zeros(2)

    def update(self, requested, wheel_delta, dt=.02):
        self.speed += .25*(np.asarray(wheel_delta)/dt-self.speed)
        requested = np.asarray(requested, dtype=float)
        if np.max(np.abs(requested)) < 1e-8:
            self.target[:] = 0
            self.integral *= .8
            return np.zeros(2)
        self.target += np.clip(requested-self.target, -25*dt, 25*dt)
        error = self.target-self.speed
        proposed = np.clip(self.integral+1.5*dt*error, -1., 1.)
        output = self.target+.25*error+proposed
        # Conditional integration prevents winding up against actuator saturation.
        self.integral = np.where((np.abs(output) < 7.8) | (output*error < 0), proposed, self.integral)
        return np.clip(self.target+.25*error+self.integral, -7.8, 7.8)


class CameraDriveCalibration:
    """Slow, bounded wheel-distance calibration from independent image-derived poses.

    Only stable, nearby observations and sufficiently long motion windows qualify.
    Fixed injected motor factors are never supplied to this class.
    """
    def __init__(self):
        self.scale = np.ones(2)
        self.anchor = None
        self.updates = 0
        self.previous_observation = None
        self.consistent_observations = 0

    def observe(self, now, pose, wheel_position, distance):
        if distance > 2. or not np.all(np.isfinite(pose)):
            self.anchor = None
            self.previous_observation = None
            self.consistent_observations = 0
            return
        if self.previous_observation is not None:
            then, old, wheels = self.previous_observation
            dl, dr = (np.asarray(wheel_position)-wheels)*WHEEL_RADIUS*self.scale
            angle = (dr-dl)/WHEEL_SEPARATION
            heading = old[2]+angle/2
            predicted = old[:2]+(dl+dr)/2*np.array([np.cos(heading), np.sin(heading)])
            consistent = (now-then < .81 and np.linalg.norm(pose[:2]-predicted) < .10
                          and abs(wrap(pose[2]-old[2]-angle)) < .12)
            self.consistent_observations = self.consistent_observations+1 if consistent else 1
            if not consistent:
                self.anchor = None
        else:
            self.consistent_observations = 1
        self.previous_observation = (now, np.array(pose), np.array(wheel_position))
        if self.consistent_observations < 3:
            return
        if self.anchor is None:
            self.anchor = (now, np.array(pose), np.array(wheel_position))
            return
        then, old, wheels = self.anchor
        elapsed = now-then
        if elapsed < 2.:
            return
        self.anchor = (now, np.array(pose), np.array(wheel_position))
        if elapsed > 4.:
            return
        angle = wrap(pose[2]-old[2])
        heading = old[2]+angle/2
        displacement = pose[:2]-old[:2]
        ds = displacement@np.array([np.cos(heading), np.sin(heading)])
        lateral = displacement@np.array([-np.sin(heading), np.cos(heading)])
        measured = (np.asarray(wheel_position)-wheels)*WHEEL_RADIUS
        if abs(lateral) > .035 or np.min(np.abs(measured)) < .20 or abs(angle) > .25:
            return
        expected = np.array([ds-angle*WHEEL_SEPARATION/2, ds+angle*WHEEL_SEPARATION/2])
        ratio = expected/measured
        if np.all((ratio > .90) & (ratio < 1.10)):
            self.scale = np.clip(.98*self.scale+.02*ratio, .90, 1.10)
            self.updates += 1


def local_motion_clear(v, omega, ranges, angles):
    """Check a one-second swept body envelope against live body-frame LiDAR hits.

    Relative obstacle checks do not need global localisation. Include LiDAR offset,
    enclosing chassis/wheels and a 1.5-cm range-noise allowance; unknown space is
    not authorised by this helper. This is a sampled engineering guard, not proof
    against every unmodelled obstacle or arbitrary measurement corruption.
    """
    ranges, angles = np.asarray(ranges), np.asarray(angles)
    if not np.all(np.isfinite(ranges)) or np.min(ranges) < .08:
        return False
    hits = ranges < 4.95
    points = np.column_stack([ranges[hits]*np.cos(angles[hits])-.049,
                              ranges[hits]*np.sin(angles[hits])])
    for t in np.linspace(0, 1., 11):
        theta = omega*t
        translation = np.array([v*t, 0.]) if abs(omega) < 1e-6 else np.array(
            [v*np.sin(theta)/omega, v*(1-np.cos(theta))/omega])
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local = (points-translation)@rotation
        inside = ((local[:, 0] > -BODY_REAR-LOCAL_NOISE_MARGIN)
                  & (local[:, 0] < BODY_FRONT+LOCAL_NOISE_MARGIN)
                  & (np.abs(local[:, 1]) < BODY_HALF_WIDTH+LOCAL_NOISE_MARGIN))
        if np.any(inside):
            return False
    return True


def recovery_grid(grid, pose):
    """Extra planning slack prevents noisy boundary cells repeatedly breaking a route."""
    from copy import copy
    from scipy.ndimage import distance_transform_edt
    from .planning import RESOLUTION
    result = copy(grid)
    result.blocked = distance_transform_edt(~grid.blocked)*RESOLUTION < .06
    cell = result.cell(pose[:2])
    return result if result.inside(cell) and not result.blocked[cell[1], cell[0]] else grid


def marker_view_route(grid, pose, visited):
    """Plan to a nearby predicted marker viewpoint using only learned occupancy.

    Visibility is a hypothesis, confirmed only by camera detection on arrival.
    Unknown cells remain provisional, just as in normal navigation.
    """
    from scipy.ndimage import distance_transform_edt
    from .planning import astar, RESOLUTION
    occupied = (grid.logodds > .5) | grid.known
    clearance = distance_transform_edt(~occupied)*RESOLUTION
    candidates = []
    stride = max(1, int(round(.20/RESOLUTION)))
    for row in range(stride, grid.blocked.shape[0]-stride, stride):
        for col in range(stride, grid.blocked.shape[1]-stride, stride):
            if grid.blocked[row, col]:
                continue
            xy = grid.xy((col, row))
            travel = np.linalg.norm(xy-pose[:2])
            if not .35 < travel < 1.6 or any(np.linalg.norm(xy-p) < .3 for p in visited):
                continue
            distance = np.linalg.norm(MARKER_XY-xy)
            if distance > np.linalg.norm(MARKER_XY-pose[:2])+.3:
                continue
            visible = True
            for edge in [-.15, 0., .15]:
                end = MARKER_XY+np.array([-.06, edge])
                for fraction in np.linspace(0, 1, max(2, int(distance/RESOLUTION))):
                    cell = grid.cell(xy+(end-xy)*fraction)
                    if not grid.inside(cell) or clearance[cell[1], cell[0]] < .04:
                        visible = False
                        break
                if not visible:
                    break
            if visible:
                candidates.append((travel+.2*distance, xy))
    routes = []
    for _, xy in sorted(candidates, key=lambda p: p[0])[:8]:
        path = astar(grid, pose[:2], xy, pose[2])
        if path is not None:
            routes.append((recovery_route_seconds(path, pose[2]), path))
    return min(routes, key=lambda p: p[0])[1] if routes else None


def inflation_escape(grid, pose, ranges, angles, target=None):
    """Leave an inflated start cell using bounded, relative LiDAR clearance.

    This does not shrink the planner's uncertainty margin. It lets a robot that is
    physically clear move slowly away from an obstacle that its global covariance
    envelope overlaps. Translation is limited by the caller's recovery budget.
    """
    if target is not None:
        delta = target['xy']-pose[:2]
        angle = wrap(np.arctan2(delta[1], delta[0])-pose[2])
        direction = target['direction']
        if direction < 0:
            angle = wrap(angle+np.pi)
        v = .04*direction if abs(angle) < .3 else 0.
        w = float(np.clip(1.4*angle, -.35, .35))
        return ((v, w) if local_motion_clear(v, w, ranges, angles) else (0., 0.)), target
    col, row = grid.cell(pose[:2])
    from scipy.ndimage import distance_transform_edt
    from .planning import RESOLUTION
    free_depth = distance_transform_edt(~grid.blocked)*RESOLUTION
    candidates = []
    radius = int(np.ceil(.48/RESOLUTION))
    for y in range(max(0, row-radius), min(grid.blocked.shape[0], row+radius+1)):
        for x in range(max(0, col-radius), min(grid.blocked.shape[1], col+radius+1)):
            if grid.blocked[y, x] or free_depth[y, x] < .08:
                continue
            delta = grid.xy((x, y))-pose[:2]
            distance = np.linalg.norm(delta)
            if .08 < distance < .45:
                angle = wrap(np.arctan2(delta[1], delta[0])-pose[2])
                direction = 1. if abs(angle) < np.pi/2 else -1.
                if direction < 0:
                    angle = wrap(angle+np.pi)
                candidates.append((distance+.05*abs(angle), angle, direction, grid.xy((x, y))))
    for _, angle, direction, xy in sorted(candidates, key=lambda p: p[0]):
        v = .04*direction if abs(angle) < .3 else 0.
        w = float(np.clip(1.4*angle, -.35, .35))
        if local_motion_clear(v, w, ranges, angles):
            return (v, w), dict(xy=xy, direction=direction)
    return (0., 0.), None


class RecoverySupervisor:
    """Continue on encoder odometry during camera absence; recover verified faults."""
    def __init__(self, camera_half_fov=None):
        self.camera_half_fov = (np.arctan(np.tan(np.deg2rad(27.5))*640/480)
                                if camera_half_fov is None else camera_half_fov)
        self.state = 'NORMAL'
        self.reason = ''
        self.entered = 0.
        self.ever_seen = False
        self.confirmations = 0
        self.last_confirmation = -1e6
        self.rejections = 0
        self.recovered = False
        self.events = []
        self.breadcrumbs = deque(maxlen=200)
        self.last_visual_position = None
        self.progress_pose = None
        self.progress_time = 0.
        self.retreat_target = None
        self.search_since = 0.
        self.view_path = None
        self.view_index = 0
        self.last_view_plan = -1e6
        self.visited_views = []
        self.aligned_since = None
        self.escape_target = None
        self.view_goal = None
        self.motion_until = 45.
        self.stop_until = 60.

    def transition(self, state, now, reason):
        if state != self.state:
            self.events.append(dict(time=float(now), event='recovery', state=state, reason=reason))
            self.state, self.entered, self.reason = state, now, reason
            if state == 'RELOCALISE':
                self.confirmations = 0
                self.retreat_target = None
                self.search_since = now
                self.view_path = None
                self.visited_views = []
                self.last_view_plan = -1e6
                self.aligned_since = None
                self.escape_target = None
                self.view_goal = None
                self.motion_until = now+45.
                self.stop_until = now+60.

    def camera_result(self, now, accepted, pose, correction=0., position_sigma=0., heading_correction=0.):
        if not accepted:
            self.rejections += 1
            if self.rejections >= 2:
                self.confirmations = 0
                self.transition('RELOCALISE', now, 'Repeated inconsistent camera observations')
            return
        self.rejections = 0
        self.ever_seen = True
        self.confirmations = self.confirmations+1 if now-self.last_confirmation < .81 else 1
        self.last_confirmation = now
        self.last_visual_position = np.asarray(pose).copy()
        if correction > .15 or heading_correction > .35:
            self.confirmations = 1
            self.transition('RELOCALISE', now, 'Large camera/odometry disagreement')
        if self.state != 'NORMAL' and self.confirmations >= 3 and position_sigma < .085:
            was_relocalising = self.state == 'RELOCALISE'
            self.transition('NORMAL', now, 'Consistent camera corrections restored')
            self.recovered = was_relocalising
            self.progress_pose = np.asarray(pose).copy()
            self.progress_time = now

    def encoder_fault(self, now):
        self.confirmations = 0
        self.transition('RELOCALISE', now, 'Implausible encoder increment')

    def budget_route(self, now, pose, path):
        # A fixed timeout must not cut off a feasible slow route before its
        # predicted travel/turn time. Keep an absolute bound on each episode.
        finish = now+1.5*recovery_route_seconds(path, pose[2])+3.
        self.motion_until = max(self.motion_until, min(self.entered+100., finish))
        self.stop_until = self.motion_until+15.

    def command(self, now, pose, covariance, camera_age, normal, ranges, angles, phase, map_grid=None, path_available=True):
        if phase == 'DOCKED':
            return normal
        if not self.breadcrumbs or np.linalg.norm(pose[:2]-self.breadcrumbs[-1][:2]) > .05:
            if self.state in ('NORMAL', 'SLOW'):
                self.breadcrumbs.append(np.asarray(pose).copy())
        sigma = np.sqrt(max(np.linalg.eigvalsh(covariance[:2, :2]).max(), 0))
        # Camera absence removes only visual corrections. Continue propagating
        # wheel odometry, retain the route and let normal LiDAR planning run.
        # No timeout or covariance threshold turns this into marker-search motion.
        if self.state == 'NORMAL' and camera_age > .8:
            self.transition('SLOW', now, 'Encoder + LiDAR navigation; no fresh camera correction')
        if self.state == 'SLOW':
            # Scale both inputs equally to retain the tracker's turning curvature.
            # Normal navigation already checks LiDAR and replans around obstacles.
            if not path_available and phase == 'NAVIGATE':
                return 0., 0.
            v, w = np.asarray(normal, dtype=float)*.75
            return (float(v), float(w)) if local_motion_clear(v, w, ranges, angles) else (0., 0.)
        if self.state == 'RELOCALISE':
            # First settle for image consistency. No covariance reset is performed.
            elapsed = now-self.entered
            if elapsed < .5:
                return 0., 0.
            if now > self.stop_until:
                return 0., 0.
            bearing = np.arctan2(*(MARKER_XY-pose[:2])[::-1])
            error = wrap(bearing-pose[2])
            v, w = 0., float(np.clip(1.4*error, -.45, .45))
            if abs(error) < .10:
                if self.aligned_since is None:
                    self.aligned_since = now
            else:
                self.aligned_since = None
            searched = self.aligned_since is not None and now-self.aligned_since > 1.5
            if abs(error) < .04:
                w = 0.
            if camera_age < .8 and abs(error) < .10:
                return 0., 0.
            if map_grid is not None and now < self.motion_until and sigma < .15:
                cell = map_grid.cell(pose[:2])
                if self.escape_target is not None:
                    if np.linalg.norm(self.escape_target['xy']-pose[:2]) < .04:
                        self.escape_target = None
                    else:
                        command, self.escape_target = inflation_escape(map_grid, pose, ranges, angles, self.escape_target)
                        return command
                if map_grid.inside(cell) and map_grid.blocked[cell[1], cell[0]]:
                    self.view_path = None
                    command, self.escape_target = inflation_escape(map_grid, pose, ranges, angles, self.escape_target)
                    return command
            if map_grid is not None and self.view_path is None and (searched or self.view_goal is not None) and now-self.last_view_plan > 2. and now < self.motion_until:
                self.last_view_plan = now
                planning_grid = recovery_grid(map_grid, pose)
                if self.view_goal is not None:
                    from .planning import astar
                    self.view_path = astar(planning_grid, pose[:2], self.view_goal, pose[2])
                if self.view_path is None:
                    self.view_path = marker_view_route(planning_grid, pose, self.visited_views)
                    # Prefer a verified viewpoint only if its route costs are
                    # comparable. Blindly returning to an old pose can create
                    # a loop when a nearer new viewpoint lies ahead.
                    if self.last_visual_position is not None and np.linalg.norm(self.last_visual_position[:2]-pose[:2]) > .12:
                        if not any(np.linalg.norm(self.last_visual_position[:2]-p) < .3 for p in self.visited_views):
                            from .planning import astar
                            previous = astar(planning_grid, pose[:2], self.last_visual_position[:2], pose[2])
                            if previous is not None and (self.view_path is None or recovery_route_seconds(previous, pose[2])
                                    < 1.1*recovery_route_seconds(self.view_path, pose[2])):
                                self.view_path = previous
                self.view_index = 1
                if self.view_path is not None:
                    self.budget_route(now, pose, self.view_path)
                    self.view_goal = self.view_path[-1].copy()
                    self.events.append(dict(time=float(now), event='marker_viewpoint',
                                            target=self.view_path[-1].tolist()))
            if self.view_path is not None:
                target = self.view_path[self.view_index]
                tolerance = .04 if self.view_index == len(self.view_path)-1 else .09
                if np.linalg.norm(target-pose[:2]) < tolerance:
                    self.view_index += 1
                    if self.view_index >= len(self.view_path):
                        self.visited_views.append(target.copy())
                        self.view_path = None
                        self.view_goal = None
                        self.aligned_since = None
                        self.search_since = now
                    return 0., 0.
                if not map_grid.clear_segment(pose[:2], target):
                    self.view_path = None
                    self.last_view_plan = now-2.1
                    return 0., 0.
                delta = target-pose[:2]
                angle = wrap(np.arctan2(delta[1], delta[0])-pose[2])
                w = float(np.clip(1.5*angle, -.45, .45))
                v = .06 if abs(angle) < .35 else 0.
                if now < self.motion_until:
                    return (v, w) if local_motion_clear(v, w, ranges, angles) else (0., 0.)
                self.view_path = None
            # Keep the retrace target until reached. Switching back to marker
            # alignment on every tick would make the two controllers fight.
            if self.retreat_target is None and searched and sigma < .15:
                while self.breadcrumbs and np.linalg.norm(self.breadcrumbs[-1][:2]-pose[:2]) < .18:
                    self.breadcrumbs.pop()
                if self.breadcrumbs:
                    self.retreat_target = self.breadcrumbs.pop()
            if self.retreat_target is not None:
                delta = self.retreat_target[:2]-pose[:2]
                if np.linalg.norm(delta) < .055:
                    self.retreat_target = None
                    self.search_since = now
                else:
                    angle = wrap(np.arctan2(delta[1], delta[0])-pose[2])
                    direction = 1 if abs(angle) < np.pi/2 else -1
                    if direction < 0:
                        angle = wrap(angle+np.pi)
                    w = float(np.clip(1.3*angle, -.35, .35))
                    v = .04*direction if abs(angle) < .3 else 0.
            if now > self.motion_until:
                # End the bounded retreat facing the reference, rather than
                # timing out with the camera pointing away from it.
                v, w = 0., float(np.clip(1.4*error, -.45, .45)) if abs(error) > .04 else 0.
            if now > self.stop_until:
                return 0., 0.
            return (v, w) if local_motion_clear(v, w, ranges, angles) else (0., 0.)
        return normal
