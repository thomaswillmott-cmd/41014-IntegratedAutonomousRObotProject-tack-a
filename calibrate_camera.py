"""Camera calibration and validation. Known poses stay in this experiment only."""
import argparse
import json
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from track_a.world import ROOT, load_world
from track_a.sensing import MarkerCamera
from track_a.estimation import PoseEKF, wrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=71)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output = ROOT/'calibration'
    output.mkdir(exist_ok=True)
    model = load_world()
    data = mujoco.MjData(model)
    camera = MarkerCamera(model, use_calibration=False)
    objects, pixels, validation = [], [], []
    try:
        for i in range(100):
            data.qpos[:3] = [rng.uniform(.9, 2.0), rng.uniform(1.1, 1.7), rng.uniform(-.02, .02)]
            roll, pitch, yaw = rng.uniform(-.15, .15, 3)
            quat = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
            data.qpos[3:7] = quat[[3, 0, 1, 2]]
            mujoco.mj_forward(model, data)
            image, measurement = camera.capture(data)
            camera.renderer.update_scene(data, camera='robot_camera')
            raw = camera.renderer.render().copy()
            corners, ids, _ = camera.detector.detectMarkers(cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY))
            if ids is not None and 0 in ids:
                index = list(ids.ravel()).index(0)
                objects.append(camera.object_points.copy())
                pixels.append(corners[index].reshape(4, 2).copy())
        K_initial = camera.K.copy()
        K_initial[0, 0] *= .9
        K_initial[1, 1] *= 1.1
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        rms, K_fit, distortion, _, _ = cv2.calibrateCamera(objects, pixels, (640, 480), K_initial, np.zeros(5), flags=flags)
        theoretical = camera.K.copy()
        camera.K = K_fit
        # Separate upright validation poses; not used for fitting.
        for x in [1.0, 1.4, 1.8, 2.15]:
            for y in [1.25, 1.4, 1.55]:
                for yaw in [-.12, 0., .12]:
                    data.qpos[:3] = [x, y, 0]
                    data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                    mujoco.mj_forward(model, data)
                    image, z = camera.capture(data)
                    if z is None:
                        continue
                    expected = PoseEKF.observation(np.array([x, y, yaw]))
                    validation.append([float(z[0]-expected[0]), float(wrap(z[1]-expected[1])),
                                       float(wrap(camera.last_yaw-yaw)), float(camera.last_reprojection)])
        residuals = np.array(validation)
        result = dict(seed=args.seed, training_views=len(objects), validation_views=len(validation),
                      reprojection_rms_px=float(rms), K=K_fit.tolist(), theoretical_K=theoretical.tolist(),
                      distortion=distortion.ravel().tolist(),
                      residual_columns=['range_m', 'bearing_rad', 'yaw_rad', 'reprojection_px'],
                      validation_bias=residuals.mean(axis=0).tolist(),
                      validation_rmse=np.sqrt(np.mean(residuals**2, axis=0)).tolist(),
                      validation_covariance=np.cov(residuals[:, :3].T).tolist(),
                      notes='Ideal pinhole simulation; zero lens distortion constrained. Tilted calibration rig; upright held-out validation. Not physical hardware calibration.')
        (output/'camera.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        np.savetxt(output/'camera_residuals.csv', residuals, delimiter=',', header=','.join(result['residual_columns']), comments='')
        cv2.imwrite(str(output/'validation_image.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(json.dumps(result, indent=2))
    finally:
        camera.close()


if __name__ == '__main__':
    main()
