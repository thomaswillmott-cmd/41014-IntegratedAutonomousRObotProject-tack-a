"""Record a genuine 1x-speed simulation and dashboard, without desktop capture.

python record_demo.py --clean --output submission/media/nominal
python record_demo.py --output submission/media/disturbed
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from track_a.dashboard import dashboard
from track_a.scenario import Scenario
from track_a.simulation import Simulation, HEADER
from track_a.trial_report import trial_report


def label(image, message, x, y, size=.65, color=(220, 230, 240)):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def recording_frame(sim, renderer, clean):
    renderer.update_scene(sim.data, camera='overview')
    path = sim.position_errors.path
    scene = renderer.scene
    for a, b in zip(path[:-1], path[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), np.array([.3, .9, .5, 1.], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .012, np.r_[a, .025], np.r_[b, .025])
        scene.ngeom += 1
    room = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    frame = np.full((1080, 1920, 3), (22, 27, 33), dtype=np.uint8)
    mode = 'NOMINAL | injected errors OFF' if clean else 'DISTURBED | all injected errors ON'
    label(frame, 'UTS 41014 | Track A | TurtleBot3 in MuJoCo', 28, 36, .85)
    label(frame, f'{mode} | seed {sim.scenario.seed} | 1x playback | t={sim.data.time:.2f}s', 28, 73, .70)
    frame[106:856, 20:1020] = cv2.resize(room, (1000, 750), interpolation=cv2.INTER_AREA)
    frame[106:906, 1030:1910] = cv2.resize(dashboard(sim), (880, 800), interpolation=cv2.INTER_AREA)
    truth = sim.true_pose_for_evaluation()
    estimate = sim.ekf.x
    sigma = np.sqrt(np.diag(sim.ekf.P))
    label(frame, 'GROUND TRUTH: evaluation/display only; controller uses EKF', 28, 899, .65, (180, 160, 240))
    label(frame, f'Actual: x={truth[0]:+.3f} m  y={truth[1]:+.3f} m  yaw={np.rad2deg(truth[2]):+.2f} deg', 28, 937, .68)
    label(frame, f'EKF:    x={estimate[0]:+.3f} m  y={estimate[1]:+.3f} m  yaw={np.rad2deg(estimate[2]):+.2f} deg', 28, 974, .68, (110, 235, 140))
    label(frame, f'Uncertainty (1 sigma): x={sigma[0]*100:.2f} cm  y={sigma[1]*100:.2f} cm  yaw={np.rad2deg(sigma[2]):.2f} deg', 28, 1011, .65)
    label(frame, 'Map: magenta = actual | green = EKF | ellipse = marginal 2-sigma axes', 1030, 940, .49)
    label(frame, f'Contacts (physics steps): {sim.collision_steps} | phase: {sim.driver.phase}', 1030, 978, .62)
    if not clean:
        label(frame, 'Push: 10-10.5s | forced camera blackout: 12-18s', 1030, 1015, .56)
    else:
        label(frame, 'Clean comparison retains real contact physics and image detection', 1030, 1015, .48)
    label(frame, 'Sensor simulation -> EKF -> estimated-state feedback -> wheel commands', 28, 1054, .62)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--duration', type=float, default=180)
    parser.add_argument('--output', type=Path, required=True, help='New output folder; includes MP4 and measured evidence')
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0 or args.seed < 0:
        parser.error('Use a positive finite duration and nonnegative seed')
    args.output.mkdir(parents=True, exist_ok=False)
    scenario = Scenario(seed=args.seed, layout='random', clean=args.clean,
                        noisy=not args.clean, dropout=not args.clean, disturbance=not args.clean)
    sim = Simulation(seed=args.seed, scenario=scenario)
    renderer = None
    writer = None
    frames = 0
    try:
        renderer = mujoco.Renderer(sim.model, height=768, width=1024)
        writer = cv2.VideoWriter(str(args.output/'simulation.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25., (1920, 1080))
        if not writer.isOpened():
            raise RuntimeError('MP4 encoder could not be opened')
        writer.write(recording_frame(sim, renderer, args.clean))
        frames += 1
        while sim.data.time < args.duration and not sim.driver.finished:
            sim.step()
            if sim.steps % 2 == 0:
                writer.write(recording_frame(sim, renderer, args.clean))
                frames += 1
            if sim.steps % 250 == 0:
                print(f'Recording: {sim.data.time:.1f}s, phase={sim.driver.phase}', flush=True)
        writer.write(recording_frame(sim, renderer, args.clean))
        frames += 1
        writer.release()
        writer = None
        summary = sim.summary()
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (args.output/'trial.csv').open('w', newline='') as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(HEADER)
            csv_writer.writerows(sim.rows)
        sim.position_errors.save(args.output)
        trial_report(args.output)
        metadata = dict(seed=args.seed, clean=args.clean, fps=25, frames=frames,
                        playback='1x; 50 Hz simulation sampled at 25 fps, with initial/final frames',
                        video_duration_s=frames/25, simulated_duration_s=float(sim.data.time),
                        docking_success=summary['docking_success'], audio='None; record your own narration',
                        source_sha256=summary['source_sha256'])
        (args.output/'recording.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata, indent=2), flush=True)
        print(f'Saved {args.output / "simulation.mp4"}', flush=True)
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        sim.close()


if __name__ == '__main__':
    main()
