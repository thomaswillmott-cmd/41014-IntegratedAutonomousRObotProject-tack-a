"""Connect the physical simulator to sensor-only estimation and control."""

import mujoco
import numpy as np
import hashlib

from .control import WaypointDriver, wheel_commands
from .estimation import PoseEKF
from .sensing import Lidar, MarkerCamera
from .world import START, ROOT, load_world
from .scenario import Scenario, GOAL
from .navigation import Navigator
from .estimation import wrap
from .evaluation import PathEvaluation
from .resilience import WheelFeedback, RecoverySupervisor, CameraDriveCalibration, MarkerRelocalizer, camera_pose, local_motion_clear

HEADER = ['time', 'true_x', 'true_y', 'true_yaw', 'est_x', 'est_y', 'est_yaw',
          'var_x', 'var_y', 'var_yaw', 'command_v', 'command_omega', 'camera_updates',
          'camera_rejected', 'lidar_min', 'waypoint_index', 'collision_steps',
          'tracking_error', 'nees', 'nis', 'camera_age', 'replans', 'tracking_active', 'estimated_tracking_error']


class Simulation:
    dt = 0.02  # Estimator and controller at 50 Hz; physics at 500 Hz.

    def __init__(self, seed=1, camera=True, automatic=True, scenario=None):
        self.scenario = scenario or Scenario(seed=seed)
        self.model = load_world(self.scenario)
        self.data = mujoco.MjData(self.model)
        self.robot_bodies = {self.model.body(name).id for name in ('base', 'wheel_left', 'wheel_right')}
        self.robot_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.robot_bodies}
        self.obstacle_geoms = {g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith('obstacle_') or self.model.geom(g).name in
                               {'crate', 'crate_2', 'barrel', 'north_wall', 'south_wall', 'east_wall', 'west_wall', 'dock_back', 'surprise_collision'}}
        self.rng = np.random.default_rng(seed)
        self.camera = MarkerCamera(self.model, pixel_noise=0. if self.scenario.clean else (.65 if self.scenario.noisy else .15), rng=self.rng) if camera else None
        self.lidar = Lidar(self.model)
        self.automatic = automatic
        self.manual = (0.0, 0.0)
        self.manual_until = 0.0
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.rows = []
        self.reset()

    def encoders(self):
        return np.array([self.data.sensor('encoder_left').data[0],
                         self.data.sensor('encoder_right').data[0]])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)
        self.ekf = PoseEKF(self.scenario.start, self.scenario.q_scale)
        if self.scenario.controller == 'baseline':
            self.driver = WaypointDriver()
        else:
            known = self.scenario.obstacles() if self.scenario.mapping == 'known' else None
            self.driver = Navigator(self.scenario.objective, self.scenario.controller, known)
        self.previous_encoder = self.encoders()
        self.wheel_loop = WheelFeedback()
        half_fov = np.arctan(320/self.camera.K[0, 0]) if self.camera is not None else None
        self.recovery = RecoverySupervisor(half_fov)
        self.drive_calibration = CameraDriveCalibration()
        self.relocalizer = MarkerRelocalizer()
        self.ranges = self.lidar.scan(self.model, self.data)
        self.measurement = None
        self.display_camera_state = 'LIVE' if self.camera is not None else 'DISABLED'
        self.display_encoder_speed = np.zeros(2)
        self.display_wheel_requested = np.zeros(2)
        if self.camera is not None:
            self.image, self.measurement = self.camera.capture(self.data)
        self.command = (0.0, 0.0)
        self.manual = (0.0, 0.0)
        self.manual_until = 0
        self.steps = 0
        self.rows = []
        self.collision_steps = 0
        self.last_camera_time = -1e6
        self.events = []
        self.nis_samples = []
        self.path_length = 0.
        self.control_effort = 0.
        self.previous_true_xy = self.true_pose_for_evaluation()[:2]
        self.last_truth = self.true_pose_for_evaluation()
        self.docking_time = None
        self.dynamic_inserted = False
        if isinstance(self.driver, Navigator):
            self.driver.observe(self.ekf.x, self.ekf.P, self.ranges, self.lidar.angles, 0.)
        self.position_errors = PathEvaluation()
        self.position_errors.observe_reference(self.driver, 0., self.ekf.x, self.last_truth)
        self.position_errors.sample(0., self.ekf.x, self.last_truth, getattr(self.driver, 'phase', 'BASELINE'), self.automatic)

    def manual_command(self, v, omega):
        self.automatic = False
        self.manual = (v, omega)
        self.manual_until = self.data.time + 2.0

    def step(self):
        previous_time = self.data.time
        self.data.xfrc_applied[:] = 0
        if self.scenario.disturbance and 10 <= previous_time < 10.5:
            self.data.xfrc_applied[self.model.body('base').id, 1] = .8
        if self.scenario.blocked and not self.dynamic_inserted and previous_time >= 8:
            heading = self.ekf.x[2]
            centre = self.ekf.x[:2] + .9*np.array([np.cos(heading), np.sin(heading)])
            self.data.mocap_pos[0] = [*centre, .25]
            self.dynamic_inserted = True
            self.events.append(dict(time=float(previous_time), event='unmapped obstacle inserted', xy=centre.tolist()))
        for _ in range(round(self.dt / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.monitor_contacts()
        if self.data.time < previous_time or not np.all(np.isfinite(self.data.qpos)):
            raise RuntimeError('Physics became unstable; stop and inspect the model settings.')
        current = self.encoders().copy()
        # Injected encoder errors are bypassed in clean comparison runs.
        if not self.scenario.clean:
            current += self.rng.normal(0, 0.002 if self.scenario.noisy else 0.0003, 2)
        wheel_delta = current-self.previous_encoder
        self.previous_encoder = current
        if np.max(np.abs(wheel_delta)) > .24:
            if self.scenario.recovery:
                self.recovery.encoder_fault(self.data.time)
            wheel_delta = np.zeros(2)
            self.ekf.P += np.diag([.02**2, .02**2, np.deg2rad(3)**2])
        self.ekf.predict(wheel_delta*self.drive_calibration.scale)
        if isinstance(self.driver, Navigator):
            self.driver.freeze_plan = self.scenario.recovery and self.recovery.state == 'RELOCALISE'
            self.driver.keep_path_without_camera = (self.scenario.recovery and not self.driver.freeze_plan
                                                    and self.data.time-self.last_camera_time > .8)
        if self.steps % 5 == 0:
            self.ranges = self.lidar.scan(self.model, self.data)
            # Keep the geometric ray scan unchanged when injected errors are off.
            if not self.scenario.clean:
                self.ranges = np.clip(self.ranges+self.rng.normal(0, .015 if self.scenario.noisy else .003, len(self.ranges)), .03, 5.)
            if isinstance(self.driver, Navigator):
                self.driver.observe(self.ekf.x, self.ekf.P, self.ranges, self.lidar.angles, self.data.time)
        if self.camera is not None and self.steps % 10 == 0:
            self.image, self.measurement = self.camera.capture(self.data)
            self.display_camera_state = 'LIVE'
            # Preserve the same short-circuit RNG draws as the original injection.
            if self.scenario.dropout:
                if 12 < self.data.time < 18:
                    self.display_camera_state = 'FORCED BLACKOUT'
                elif self.rng.random() < .25:
                    self.display_camera_state = 'DROPPED FRAME'
                if self.display_camera_state != 'LIVE':
                    self.measurement = None
                    self.image = np.zeros_like(self.image)

            if self.measurement is not None:
                before = self.ekf.x.copy()
                accepted = self.ekf.update(self.measurement, self.camera.last_yaw)
                self.nis_samples.append(dict(time=float(self.data.time), nis=self.ekf.last_nis, accepted=bool(accepted)))
                if self.scenario.recovery:
                    self.recovery.camera_result(self.data.time, accepted, self.ekf.x,
                                                np.linalg.norm(self.ekf.x[:2]-before[:2]),
                                                np.sqrt(max(np.linalg.eigvalsh(self.ekf.P[:2, :2]).max(), 0)),
                                                abs(wrap(self.ekf.x[2]-before[2])))
                    if self.recovery.state == 'RELOCALISE':
                        candidate = self.relocalizer.observe(self.data.time, self.measurement,
                                                             self.camera.last_yaw, wheel_delta/self.dt)
                        if candidate is not None and not accepted:
                            self.ekf.x, self.ekf.P = candidate
                            self.events.append(dict(time=float(self.data.time), event='Camera relocalisation from five consistent observations'))
                            self.recovery.rejections = 0
                            self.recovery.confirmations = 0
                            # Discard stale learned occupancy after a global pose correction;
                            # retain configured known geometry. Next scans rebuild evidence.
                            if isinstance(self.driver, Navigator):
                                self.driver.map.logodds[:] = 0
                                self.driver.map.observed[:] = False
                if accepted:
                    if isinstance(self.driver, Navigator) and (np.linalg.norm(self.ekf.x[:2]-before[:2]) > .15
                            or abs(wrap(self.ekf.x[2]-before[2])) > .35):
                        self.driver.map.logodds[:] = 0
                        self.driver.map.observed[:] = False
                        self.driver.last_plan = -10.
                    self.last_camera_time = self.data.time
                    if self.scenario.camera_calibration and self.camera.last_yaw is not None and self.recovery.state == 'NORMAL':
                        self.drive_calibration.observe(self.data.time, camera_pose(self.measurement, self.camera.last_yaw),
                                                       current, self.measurement[0])
        if self.automatic:
            if isinstance(self.driver, Navigator):
                self.command = self.driver.command(self.ekf.x.copy(), self.ranges.copy(), self.lidar.angles,
                                                   self.data.time-self.last_camera_time)
            else:
                self.command = self.driver.command(self.ekf.x.copy(), self.ranges.copy(), self.lidar.angles)
        else:
            self.command = self.manual if self.data.time < self.manual_until else (0.0, 0.0)
        if self.automatic and self.scenario.recovery and isinstance(self.driver, Navigator):
            self.command = self.recovery.command(self.data.time, self.ekf.x, self.ekf.P,
                self.data.time-self.last_camera_time, self.command, self.ranges, self.lidar.angles, self.driver.phase, self.driver.map,
                len(self.driver.path) >= 2 or self.driver.phase != 'NAVIGATE')
            if self.recovery.state != 'NORMAL':
                self.driver.status = f'{self.recovery.state}: {self.recovery.reason}'
            if self.recovery.recovered:
                # Rebuild scan evidence in the corrected frame instead of mixing
                # pre-correction obstacle stamps with the recovered pose.
                self.driver.map.logodds[:] = 0
                self.driver.map.observed[:] = False
                self.driver.last_plan = -10.
                self.driver.freeze_plan = False
                self.recovery.recovered = False
        # Compact global planning does not cover turning corners. Keep the actual
        # chassis/wheel footprint in the live LiDAR check for EVERY auto command.
        if self.automatic and isinstance(self.driver, Navigator) and any(abs(v) > 1e-8 for v in self.command):
            if not local_motion_clear(*self.command, self.ranges, self.lidar.angles):
                self.command = (0., 0.)
                self.driver.status = 'Local body clearance stop'
        requested = wheel_commands(*self.command)/self.drive_calibration.scale
        # UI diagnostics only: no input from these values returns to control.
        self.display_wheel_requested = requested.copy()
        self.display_encoder_speed += .25*(wheel_delta/self.dt-self.display_encoder_speed)
        # Feedback reads the same noisy encoder increments; no exact speed sensor.
        self.data.ctrl[:] = self.wheel_loop.update(requested, wheel_delta, self.dt) if self.scenario.wheel_feedback else requested
        if self.scenario.noisy:
            self.data.ctrl[:] *= [.97, 1.02]
        if self.driver.finished and self.docking_time is None:
            self.docking_time = float(self.data.time)
        self.steps += 1
        self.record_evaluation()

    def true_pose_for_evaluation(self):
        # Truth is restricted to evaluation/display. Neither EKF nor driver calls this.
        rotation = self.data.body('base').xmat.reshape(3, 3)
        return np.array([*self.data.body('base').xpos[:2], np.arctan2(rotation[1, 0], rotation[0, 0])])

    def record_evaluation(self):
        truth = self.true_pose_for_evaluation()
        self.path_length += float(np.linalg.norm(truth[:2]-self.previous_true_xy))
        self.previous_true_xy = truth[:2]
        self.last_truth = truth
        self.position_errors.observe_reference(self.driver, self.data.time, self.ekf.x, truth)
        self.control_effort += self.dt*(self.command[0]**2+self.command[1]**2)
        if self.steps % 5 == 0:
            scored = self.automatic and (self.docking_time is None or self.data.time <= self.docking_time)
            self.position_errors.sample(self.data.time, self.ekf.x, truth,
                                        getattr(self.driver, 'phase', 'BASELINE'), scored)
            error = self.ekf.x-truth
            error[2] = wrap(error[2])
            nees = float(error@np.linalg.solve(self.ekf.P, error))
            reference = getattr(self.driver, 'tracking_reference', None)
            tracking_error = 0.
            if reference is not None:
                a, b = reference
                tangent = (b-a)/max(1e-9, np.linalg.norm(b-a))
                tracking_error = float((truth[:2]-a)@np.array([-tangent[1], tangent[0]]))
            self.rows.append([self.data.time, *truth, *self.ekf.x, *np.diag(self.ekf.P),
                              *self.command, self.ekf.updates, self.ekf.rejected,
                              self.ranges.min(), self.driver.index, self.collision_steps,
                              tracking_error, nees, self.ekf.last_nis,
                              min(999., self.data.time-self.last_camera_time), getattr(self.driver, 'replans', 0),
                              float(reference is not None), getattr(self.driver, 'tracking_error', 0)])

    def monitor_contacts(self):
        """Evaluation only, at every 500-Hz physics step, including brief contacts."""
        for contact in self.data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if ((g1 in self.robot_geoms and g2 in self.obstacle_geoms) or
                    (g2 in self.robot_geoms and g1 in self.obstacle_geoms)):
                self.collision_steps += 1
                break

    def summary(self):
        rows = np.asarray(self.rows)
        truth = self.true_pose_for_evaluation()
        position_error = rows[:, 1:3]-rows[:, 4:6]
        goal_distance = float(np.linalg.norm(truth[:2]-GOAL[:2]))
        heading_error = float(abs(wrap(truth[2]-GOAL[2])))
        fingerprint = hashlib.sha256()
        for path in sorted((ROOT/'track_a').glob('*.py')):
            fingerprint.update(path.name.encode())
            fingerprint.update(path.read_bytes())
        calibration = ROOT/'calibration'/'camera.json'
        if calibration.exists():
            fingerprint.update(calibration.read_bytes())
        return dict(seed=self.scenario.seed, scenario=self.scenario.as_dict(), source_sha256=fingerprint.hexdigest(),
                    simulated_seconds=float(self.data.time), route_complete=bool(self.driver.finished),
                    docking_success=bool(self.driver.finished and goal_distance < .08 and heading_error < .12 and self.collision_steps == 0),
                    docking_time_s=self.docking_time, path_length_m=self.path_length,
                    position_rmse_m=float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
                    final_true_goal_distance_m=goal_distance, final_heading_error_rad=heading_error,
                    tracking_rmse_m=float(np.sqrt(np.mean(rows[rows[:, 22] > 0, 17]**2))) if np.any(rows[:, 22] > 0) else None,
                    active_path_rmse_m=self.position_errors.tracking_rmse,
                    continuous_estimation_rmse_m=self.position_errors.estimation_rmse,
                    average_position_error_m=self.position_errors.average_position_error,
                    path_versions=len(self.position_errors.history),
                    recovery_state=self.recovery.state, recovery_events=self.recovery.events,
                    wheel_distance_scale=self.drive_calibration.scale.tolist(),
                    camera_calibration_updates=self.drive_calibration.updates,
                    planner_clearance=getattr(self.driver, 'clearance', None),
                    mean_nees=float(np.mean(rows[:, 18])), nees_95_coverage=float(np.mean(rows[:, 18] < 7.815)),
                    control_effort=self.control_effort, camera_enabled=self.camera is not None,
                    camera_updates=self.ekf.updates, camera_rejected=self.ekf.rejected,
                    replans=getattr(self.driver, 'replans', 0), collision_control_steps=self.collision_steps,
                    collision_physics_steps=self.collision_steps,
                    final_estimate=self.ekf.x.tolist(), final_truth=truth.tolist(), events=self.events)

    def close(self):
        if self.camera is not None:
            self.camera.close()
