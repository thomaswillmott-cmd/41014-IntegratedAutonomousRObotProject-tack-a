"""Assemble the silent review edit from genuine record_demo.py clips.

Run from the project root: python submission/build_video_edit.py
Add your own narration in a video editor; no audio is generated here.
"""
import json
from pathlib import Path
import textwrap

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT/'submission'/'media'
FPS = 25
SIZE = (1920, 1080)


def text(frame, message, x, y, scale=.9, color=(225, 232, 240)):
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def card(title, lines, heading='UTS 41014 | Track A | Thomas Willmott'):
    frame = np.full((1080, 1920, 3), (25, 30, 36), np.uint8)
    text(frame, heading, 85, 70, .85, (130, 210, 170))
    text(frame, title, 85, 155, 1.35)
    cv2.line(frame, (85, 190), (1835, 190), (70, 100, 95), 2)
    y = 280
    for line in lines:
        for wrapped in textwrap.wrap(line, width=95):
            text(frame, wrapped, 95, y, .9)
            y += 52
        y += 28
    text(frame, 'Ground truth is evaluation only. Controller uses the EKF estimate.', 85, 1015, .72, (155, 170, 185))
    return frame


def graph_card(title, path):
    frame = card(title, [])
    chart = cv2.imread(str(path))
    if chart is None:
        raise FileNotFoundError(path)
    scale = min(1750/chart.shape[1], 755/chart.shape[0])
    chart = cv2.resize(chart, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y, x = 220, (1920-chart.shape[1])//2
    frame[y:y+chart.shape[0], x:x+chart.shape[1]] = chart
    return frame


def main():
    destination = MEDIA/'review_edit.mp4'
    if destination.exists():
        raise FileExistsError(f'Preserve or rename the existing edit first: {destination}')
    nominal = json.loads((MEDIA/'nominal_seed22/recording.json').read_text())
    disturbed = json.loads((MEDIA/'disturbed_seed22/recording.json').read_text())
    if not nominal['docking_success'] or not disturbed['docking_success']:
        raise RuntimeError('Review unsuccessful recordings before assembling the planned success demonstration')
    segments = [
        ('Task and contribution', 20, card('Visual waypoint navigation and docking', [
            'TurtleBot3 in MuJoCo | LiDAR obstacle discovery | ArUco-marked station',
            'Thomas Willmott | SID 25494826 | individual implementation',
            'Encoders + camera -> EKF -> estimated-state feedback -> wheel commands',
            'AI assistance: ChatGPT/Codex for implementation, debugging and documentation.',
            'This edit uses genuine runs. Simulation playback is 1x.'
        ])),
        ('Model, sensing and frames', 35, card('Model, sensors and coordinate frames', [
            'State: [x, y, heading]. Wheel radius 0.033m; axle separation 0.288m.',
            'x_dot = v*cos(heading); y_dot = v*sin(heading); heading_dot = omega.',
            'World: floor x/y, z up. Robot: x forward, y left. Camera: right/down/forward.',
            'LiDAR maps obstacles using estimated pose; camera PnP corrects range, bearing and yaw.',
            'Calibration: 100 rendered views; 0.037px fit RMS; 7.44mm held-out range RMSE.',
            'Local controllability rank: 3 when moving, 2 at rest. Full camera measurement rank: 3.'
        ])),
        ('Nominal run', None, MEDIA/'nominal_seed22/simulation.mp4'),
        ('Estimator and controller', 30, card('Prediction, correction and feedback', [
            'EKF prediction: wheel increments at 50Hz; process covariance allows model error.',
            'Camera updates: distance-dependent covariance and innovation gating.',
            'Scheduled LQR: Q = diag(6,14,3), R = diag(2,1). Lateral error is penalised more.',
            'At 0.18m/s, local eigenvalues are approximately 0.966, 0.967 and 0.992.',
            'This local result excludes saturation, replanning and mode switching.',
            'Final docking uses separate pose feedback and requires fresh camera corrections.'
        ])),
        ('Disturbed run', None, MEDIA/'disturbed_seed22/simulation.mp4'),
        ('Repeated-trial results', 35, card('Measured results: three seeded layouts, all errors on', [
            'Seeds: 524882, 407433, 547246. All 3 docked; no recorded obstacle contacts.',
            'Completion times: 39.40s, 39.04s and 46.68s.',
            'Mean XY estimation errors: 0.96cm, 1.03cm and 1.42cm.',
            'Active-path tracking RMSE: 3.71cm, 3.43cm and 4.75cm.',
            'Final dock distance: 4.82-4.93cm. Heading error: 0.30-1.59 degrees.',
            'These are distinct measurements; three trials do not establish universal reliability.'
        ])),
        ('Tracking and reference changes', 25, graph_card('Tracking error differs from estimation error', MEDIA/'disturbed_seed22/tracking_errors.png')),
        ('Limits and reproducibility', 30, card('Limits, evidence and reproducibility', [
            'Known initial/station poses, restricted layouts, ideal simulated camera; no full SLAM.',
            'No hardware validation or electrical charging contact. Gains are not proven optimal.',
            'Camera losses remove corrections; encoder drift and uncertainty can accumulate.',
            'python run_robot.py --layout random --seed 22 | python run_batch.py',
            'Fresh Windows environment: all 51 tests passed. Code, telemetry and plots are in GitHub.',
            'Sources: ROBOTIS model (Apache-2.0), MuJoCo, OpenCV and SciPy. AI use declared.'
        ])),
    ]
    total_seconds = sum(duration if duration is not None else
                        (nominal if 'Nominal' in name else disturbed)['video_duration_s']
                        for name, duration, _ in segments)
    if total_seconds >= 300:
        raise RuntimeError(f'Edit exceeds the five-minute limit: {total_seconds}')
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError('Video encoder unavailable')
    timeline = []
    count = 0
    try:
        for name, duration, source in segments:
            start = count/FPS
            print(f'{start:.2f}s: {name}', flush=True)
            if duration is None:
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise RuntimeError(f'Cannot open clip: {source}')
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        if frame.shape[:2] != (1080, 1920):
                            raise ValueError('Unexpected clip dimensions')
                        writer.write(frame)
                        count += 1
                finally:
                    capture.release()
            else:
                for _ in range(round(duration*FPS)):
                    writer.write(source)
                    count += 1
            timeline.append(dict(section=name, start_s=start, end_s=count/FPS))
    finally:
        writer.release()
    (MEDIA/'review_timeline.json').write_text(json.dumps(dict(duration_s=count/FPS,
        audio='Silent edit: add student narration before submission', sections=timeline), indent=2))
    print(f'Saved {destination}; duration {count/FPS:.2f}s', flush=True)


if __name__ == '__main__':
    main()
