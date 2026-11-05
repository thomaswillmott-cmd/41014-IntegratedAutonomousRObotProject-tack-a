"""Run the TurtleBot3 room demo. Try: python run_robot.py --help"""

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from track_a.dashboard import TITLE, dashboard
from track_a.simulation import Simulation, HEADER
from track_a.world import ROOT, WAYPOINTS
from track_a.scenario import Scenario
from track_a.path_views import selected_path, step_path



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Run fast without windows; camera still renders')
    parser.add_argument('--exit-on-complete', action='store_true', help='Close the viewer when the autonomous mission finishes (for batches)')
    parser.add_argument('--manual', action='store_true', help='Start stopped in manual mode')
    parser.add_argument('--paused', action='store_true', help='Open paused; press P to begin')
    parser.add_argument('--no-camera', action='store_true', help='Encoder-only development run; no ArUco updates')
    parser.add_argument('--no-dashboard', action='store_true', help='Show only the MuJoCo viewer')
    parser.add_argument('--duration', type=float, default=180, help='Maximum simulated seconds (default 180)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--layout', choices=['fixed', 'random'], default='fixed')
    parser.add_argument('--objective', choices=['distance', 'time'], default='time')
    parser.add_argument('--map', choices=['lidar', 'known'], default='lidar', dest='mapping')
    parser.add_argument('--controller', choices=['lqr', 'proportional', 'baseline'], default='lqr')
    parser.add_argument('--clean', action='store_true', help='Disable all injected errors for comparison')
    parser.add_argument('--all-errors', action='store_true', help='Enable all sensor/stress errors (the default)')
    parser.add_argument('--sensor-noise', action='store_true', help='Restore normal injected encoder/LiDAR/corner noise')
    parser.add_argument('--noisy', action='store_true', help='Stronger sensor noise and unequal wheel drive')
    parser.add_argument('--dropout', action='store_true', help='Camera blackout at 12-18s plus random lost frames')
    parser.add_argument('--disturbance', action='store_true', help='Physical sideways force at 10s')
    parser.add_argument('--blocked', action='store_true', help='Insert an unmapped obstacle ahead at 8s')
    parser.add_argument('--no-recovery', action='store_true', help='Disable recovery supervisor for comparison')
    parser.add_argument('--open-loop-wheels', action='store_true', help='Disable encoder speed feedback for comparison')
    parser.add_argument('--no-camera-calibration', action='store_true', help='Disable slow image-based drive calibration')
    parser.add_argument('--start-y', type=float, default=-1.5)
    parser.add_argument('--q-scale', type=float, default=1., help='Slip process covariance multiplier; 0 for ablation')
    parser.add_argument('--start-yaw', type=float, default=0., help='Starting heading in degrees')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'latest')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    error_options = args.all_errors or args.sensor_noise or args.noisy or args.dropout or args.disturbance
    if args.clean and error_options:
        parser.error('--clean cannot be combined with injected-error options')
    # No explicit mode: run the complete injected-error experiment.
    if args.all_errors or not (args.clean or error_options):
        args.noisy = args.dropout = args.disturbance = True
    clean = args.clean
    args.output.mkdir(parents=True, exist_ok=True)
    scenario = Scenario(seed=args.seed, layout=args.layout, objective=args.objective, mapping=args.mapping, clean=clean,
                        controller=args.controller, noisy=args.noisy, dropout=args.dropout,
                        disturbance=args.disturbance, blocked=args.blocked, q_scale=args.q_scale,
                        recovery=not args.no_recovery, wheel_feedback=not args.open_loop_wheels,
                        camera_calibration=not args.no_camera_calibration,
                        start=np.array([-2., args.start_y, np.deg2rad(args.start_yaw)]))
    sim = Simulation(seed=args.seed, camera=not args.no_camera, automatic=not args.manual, scenario=scenario)
    sim.display_path_id = None
    pending_keys = []
    paused = args.paused
    if args.headless and paused:
        parser.error('--paused requires the interactive viewer')

    def handle(key):
        nonlocal paused
        if key in (ord('W'), ord('w')):
            sim.manual_command(0.16, 0)
        elif key in (ord('S'), ord('s')):
            sim.manual_command(-0.12, 0)
        elif key in (ord('A'), ord('a')):
            sim.manual_command(0, 0.8)
        elif key in (ord('D'), ord('d')):
            sim.manual_command(0, -0.8)
        elif key == 32:
            sim.manual_command(0, 0)
        elif key in (ord('M'), ord('m')):
            sim.automatic = not sim.automatic
            sim.manual = (0, 0)
        elif key in (ord('R'), ord('r')):
            sim.reset()
            sim.display_path_id = None
        elif key in (ord('['), ord(']')):
            sim.display_path_id = step_path(sim.position_errors.history, sim.display_path_id,
                                            -1 if key == ord('[') else 1)
        elif key == ord('0'):
            sim.display_path_id = None
        elif key in (ord('P'), ord('p')):
            paused = not paused

    print('TurtleBot3 Track A demo | model: ROBOTIS Waffle Pi')
    print('Injected errors: OFF (clean comparison)' if clean else 'Injected errors: ON (experimental sensor/stress model)')
    print('W/A/S/D drive | SPACE stop | M auto/manual | R reset | P pause | [ / ] browse paths | 0 latest')
    print(f'{args.controller.upper()} | {args.mapping} map | {args.objective} objective | ground truth: evaluation only')
    viewer = None
    last_dashboard = -1
    try:
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=pending_keys.append,
                                                  show_left_ui=False, show_right_ui=False)
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 7.5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -60
            if not args.no_dashboard:
                cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(TITLE, 1050, 916)
        while sim.data.time < args.duration:
            start = time.perf_counter()
            while pending_keys:
                handle(pending_keys.pop(0))
            if not paused:
                sim.step()
            if args.exit_on_complete and sim.driver.finished and sim.automatic:
                break
            if viewer is not None:
                if not viewer.is_running():
                    break
                if hasattr(sim, 'position_errors'):
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        history = sim.position_errors.history
                        selected = selected_path(history, sim.display_path_id)
                        versions = [v for v in history if v['path_id'] in (selected, sim.position_errors.path_id)]
                        for version in versions:
                            active = version['path_id'] == sim.position_errors.path_id
                            path = np.asarray(version['points']).reshape(-1, 2)
                            for a, b in zip(path[:-1], path[1:]):
                                index = viewer.user_scn.ngeom
                                if index >= viewer.user_scn.maxgeom:
                                    break
                                geom = viewer.user_scn.geoms[index]
                                rgba = np.array([.39, .86, .59, .9] if active else [.96, .75, .20, .8], dtype=np.float32)
                                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                   np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
                                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .008 if active else .004,
                                                    np.r_[a, .018 if active else .012], np.r_[b, .018 if active else .012])
                                viewer.user_scn.ngeom += 1
                viewer.sync()
                if not args.no_dashboard:
                    if paused or sim.steps // 5 != last_dashboard:
                        cv2.imshow(TITLE, dashboard(sim, paused))
                        last_dashboard = sim.steps // 5
                    key = cv2.waitKey(1)
                    if key == 27 or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key >= 0:
                        handle(key)
                time.sleep(max(0, (0.05 if paused else sim.dt) - (time.perf_counter() - start)))
            elif sim.driver.finished and sim.automatic:
                break
        if sim.rows:
            with (args.output/'trial.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADER)
                writer.writerows(sim.rows)
            summary = sim.summary()
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            (args.output/'innovations.json').write_text(json.dumps(sim.nis_samples, indent=2), encoding='utf-8')
            sim.position_errors.save(args.output)
            from plot_results import plot_position_errors
            plot_position_errors(args.output)
            cv2.imwrite(str(args.output/'dashboard.png'), dashboard(sim))
            if sim.camera is not None:
                sim.camera.renderer.update_scene(sim.data, camera='overview')
                scene = sim.camera.renderer.render()
                cv2.imwrite(str(args.output/'scene.png'), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
            print(json.dumps(summary, indent=2))
            print(f'Results saved to: {args.output}')
    finally:
        sim.data.ctrl[:] = 0
        if viewer is not None:
            viewer.close()
        sim.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
