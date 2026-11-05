"""Simulated LiDAR rays and actual image-based ArUco detection."""

import cv2
import json
import mujoco
import numpy as np

from .world import MARKER_SIZE, ROOT
from .estimation import wrap


class Lidar:
    def __init__(self, model):
        self.site = model.site('lidar').id
        self.base = model.body('base').id
        self.angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        self.groups = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)

    def scan(self, model, data):
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        distances = np.empty(len(self.angles))
        geom_id = np.zeros(1, dtype=np.int32)
        for i, angle in enumerate(self.angles):
            ray = rotation @ np.array([np.cos(angle), np.sin(angle), 0.0])
            distance = mujoco.mj_ray(model, data, origin, ray, self.groups, 1, self.base, geom_id)
            distances[i] = min(distance, 5.0) if distance >= 0 else 5.0
        return distances


class MarkerCamera:
    def __init__(self, model, width=640, height=480, pixel_noise=0., rng=None, use_calibration=True):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[model.camera('robot_camera').id]) / 2))
        self.K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=float)
        self.distortion = np.zeros(5)  # Ideal simulated pinhole camera.
        self.calibrated = False
        calibration_path = ROOT/'calibration'/'camera.json'
        if use_calibration and calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
            self.K = np.asarray(calibration['K'], dtype=float)
            self.K[0, :] *= width/640
            self.K[1, :] *= height/480
            self.distortion = np.asarray(calibration['distortion'], dtype=float)
            self.calibrated = True
        self.pixel_noise = pixel_noise
        self.rng = rng or np.random.default_rng(1)
        self.last_yaw = None
        self.last_reprojection = None
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        h = MARKER_SIZE / 2
        self.object_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)

    def capture(self, data):
        self.renderer.update_scene(data, camera='robot_camera')
        image = self.renderer.render().copy()
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        measurement = None
        self.last_yaw = None
        self.last_reprojection = None
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            for points, marker_id in zip(corners, ids.ravel()):
                if marker_id != 0:
                    continue
                pixels = points.reshape(4, 2).astype(np.float64)
                if self.pixel_noise > 0:
                    pixels += self.rng.normal(0, self.pixel_noise, pixels.shape)
                ok, rotation, translation = cv2.solvePnP(self.object_points, pixels,
                                                 self.K, self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok and translation[2, 0] > 0:
                    projected, _ = cv2.projectPoints(self.object_points, rotation, translation, self.K, self.distortion)
                    self.last_reprojection = float(np.sqrt(np.mean((projected.reshape(4, 2)-pixels)**2)))
                    if self.last_reprojection > 2.5:
                        continue
                    right, _, forward = translation.ravel()
                    measurement = np.array([np.hypot(right, forward), np.arctan2(-right, forward)])
                    R_cm, _ = cv2.Rodrigues(rotation)
                    R_wm = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
                    R_wc = R_wm @ R_cm.T
                    self.last_yaw = wrap(np.arctan2(R_wc[1, 2], R_wc[0, 2]))
        return image, measurement

    def close(self):
        self.renderer.close()
