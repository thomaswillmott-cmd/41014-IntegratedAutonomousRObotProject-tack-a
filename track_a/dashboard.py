"""Sensor and results display only; nothing here feeds control."""

import cv2
import numpy as np

from .path_views import selected_path
from .display_metrics import wheel_slip_for_evaluation

TITLE = 'Track A sensors - camera / LiDAR / estimated pose'


def dashboard(sim, paused=False):
    evaluation = sim.position_errors
    selected = selected_path(evaluation.history, getattr(sim, 'display_path_id', None))
    gold, grey = (50, 190, 245), (130, 135, 140)
    canvas = np.full((1000, 1100, 3), (25, 30, 36), np.uint8)
    canvas[:480, :640] = cv2.cvtColor(sim.image, cv2.COLOR_RGB2BGR)

    def text(message, x, y, scale=0.55, color=(210, 220, 230)):
        cv2.putText(canvas, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def point(xy):
        return (int(870 + xy[0]*65), int(255 - xy[1]*65))

    text('MAP | truth is shown for evaluation only', 655, 25, 0.47)
    if hasattr(sim.driver, 'clearance'):
        budget = sim.driver.clearance
        text(f"Clearance {100*budget['total_m']:.1f}cm = plan {100*budget['planning_radius_m']:.1f} + uncertainty {100*budget['uncertainty_m']:.1f} + grid {100*budget['grid_allowance_m']:.1f}",
             650, 48, .35)
    cv2.rectangle(canvas, point([-3, 2.5]), point([3, -2.5]), (100, 115, 130), 1)
    for _, kind, centre, size in sim.scenario.obstacles():
        if kind == 'cylinder':
            cv2.circle(canvas, point(centre), int(size[0]*65), (70, 100, 170), -1)
        else:
            cv2.rectangle(canvas, point(np.array(centre)+[-size[0], size[1]]),
                          point(np.array(centre)+[size[0], -size[1]]), (70, 100, 170), -1)
    waypoints = evaluation.path
    for version in evaluation.history:
        old = np.array([point(p) for p in version['points']], np.int32)
        if len(old) > 1 and version['path_id'] == selected:
            cv2.polylines(canvas, [old], False, gold, 2)
    if hasattr(sim.driver, 'map'):
        grid = sim.driver.map
        for row, col in np.argwhere(grid.logodds > .5):
            cv2.circle(canvas, point(grid.xy((col, row))), 2, (100, 190, 240), -1)
    route = np.array([point(p) for p in waypoints], np.int32)
    if len(route) > 1:
        cv2.polylines(canvas, [route], False, (100, 220, 150), 2)
    if sim.rows:
        for indices, color in [((1, 2), (210, 100, 220)), ((4, 5), (90, 220, 110))]:
            trail = np.array([point([r[indices[0]], r[indices[1]]]) for r in sim.rows[::3]], np.int32)
            if len(trail) > 1:
                cv2.polylines(canvas, [trail], False, color, 1)
    for i, p in enumerate(waypoints):
        cv2.circle(canvas, point(p), 5, (100, 220, 150), 1)
        text(str(i+1), point(p)[0]+7, point(p)[1], 0.4)
    truth = sim.true_pose_for_evaluation()
    metrics = evaluation.measure(sim.data.time, sim.ekf.x, truth, getattr(sim.driver, 'phase', 'BASELINE'))
    if metrics['reference_x'] is not None:
        reference = point([metrics['reference_x'], metrics['reference_y']])
        cv2.line(canvas, point(truth[:2]), reference, (0, 165, 255), 2)
        cv2.circle(canvas, reference, 4, (100, 220, 230), -1)
    for pose, color in [(truth, (210, 100, 220)), (sim.ekf.x, (80, 240, 100))]:
        centre = point(pose[:2])
        end = point(pose[:2] + 0.28*np.array([np.cos(pose[2]), np.sin(pose[2])]))
        cv2.circle(canvas, centre, 6, color, 2)
        cv2.line(canvas, centre, end, color, 2)
    values, vectors = np.linalg.eigh(sim.ekf.P[:2, :2])
    axes = tuple(max(1, int(2*np.sqrt(max(v, 0))*65)) for v in values[::-1])
    axis = vectors[:, -1]
    cv2.ellipse(canvas, point(sim.ekf.x[:2]), axes, float(-np.rad2deg(np.arctan2(axis[1], axis[0]))),
                0, 360, (60, 140, 80), 1)
    text(f'Green: active route | Gold: selected Path {selected}', 655, 450, 0.43)
    text('Green: EKF | Orange link: actual-to-path error', 655, 470, 0.43)
    camera_state = getattr(sim, 'display_camera_state', 'LIVE')
    if camera_state == 'LIVE':
        text('ROBOT CAMERA | real rendered image + ArUco detection', 12, 25, .52, (60, 235, 140))
    else:
        canvas[:480, :640] = 0
        text('ROBOT CAMERA | signal unavailable', 12, 25, .52, (130, 130, 150))
        text('CAMERA DISABLED' if camera_state == 'DISABLED' else 'CAMERA ERROR', 145, 220, 1., (80, 80, 245))
        subtitle = {'FORCED BLACKOUT': 'Simulated camera blackout',
                    'DROPPED FRAME': 'Simulated frame loss',
                    'DISABLED': 'Camera not enabled for this run'}.get(camera_state, camera_state)
        text(subtitle, 145, 260, .62)
        text('Waiting for the next available frame...' if camera_state != 'DISABLED' else 'Encoder-based navigation only',
             125, 300, .48, (145, 150, 160))
    origin = (125, 600)
    cv2.circle(canvas, origin, 90, (65, 75, 85), 1)
    for a, r in zip(sim.lidar.angles, sim.ranges):
        p = (int(origin[0]-np.sin(a)*min(r, 5)*18), int(origin[1]-np.cos(a)*min(r, 5)*18))
        cv2.circle(canvas, p, 1, (220, 210, 70), -1)
    cv2.arrowedLine(canvas, origin, (125, 575), (80, 230, 100), 2)
    text('LiDAR | up = robot forward', 15, 710, 0.42)
    mode = 'AUTO' if sim.automatic else 'MANUAL'
    text(f'{mode}   {"PAUSED" if paused else "RUNNING"}   t={sim.data.time:.1f}s', 255, 510, .62)
    message = sim.driver.status if sim.automatic else 'Drive with W A S D; SPACE stops'
    words, lines, line = message.split(), [], ''
    for word in words:
        candidate = f'{line} {word}'.strip()
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0] > 465 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    for i, line in enumerate(lines[:2]):
        text(line, 255, 536+20*i, .45)
    x, y, theta = sim.ekf.x
    text(f'EKF: x={x:+.2f}m  y={y:+.2f}m  heading={np.rad2deg(theta):+.1f}deg', 255, 581, .44)
    error = np.linalg.norm(truth[:2]-sim.ekf.x[:2])
    average = evaluation.average_position_error
    average_text = f'{average*100:.2f} cm' if average is not None else 'n/a'
    text(f'Position error: {error*100:.2f} cm | Avg: {average_text}', 255, 607, .53)
    marker = 'available' if sim.measurement is not None else 'unavailable'
    text(f'Marker: {marker} | LiDAR nearest: {sim.ranges.min():.2f}m', 255, 634, .44)

    # Evaluation display only.
    cv2.rectangle(canvas, (748, 489), (1090, 668), (39, 45, 53), -1)
    scenario = sim.scenario
    enabled = not scenario.clean
    sigma_encoder = (.002 if scenario.noisy else .0003) if enabled else 0.
    sigma_lidar = (1.5 if scenario.noisy else .3) if enabled else 0.
    sigma_camera = (.65 if scenario.noisy else .15) if enabled else 0.
    text('INJECTED ERRORS | '+('ON' if enabled else 'OFF'), 760, 510, .44, gold)
    text(f'Encoder noise: {sigma_encoder:g} rad sigma', 760, 531, .39)
    text(f'LiDAR: {sigma_lidar:g} cm | Camera: {sigma_camera:g} px sigma', 760, 549, .37)
    text('Drive bias: L -3% | R +2%' if scenario.noisy else 'Drive bias: OFF', 760, 567, .39)
    push_active = bool(np.any(sim.data.xfrc_applied[sim.model.body('base').id, :3]))
    push = '0.8 N ACTIVE' if push_active else (('0.8 N ready (10s)' if sim.data.time < 10 else '0.8 N completed') if scenario.disturbance else 'OFF')
    text('Sideways push: '+push, 760, 585, .39, (80, 150, 255) if push_active else (210, 220, 230))
    text('Camera loss: 25% frames + 12-18s blackout' if scenario.dropout else 'Camera loss injection: OFF', 760, 603, .35)
    mismatch = sim.display_wheel_requested-sim.display_encoder_speed
    text(f'Speed error L/R: {mismatch[0]:+.2f}/{mismatch[1]:+.2f} rad/s', 760, 621, .37)
    slip = wheel_slip_for_evaluation(sim)*100
    text(f'Rolling slip L/R: {slip[0]:+.2f}/{slip[1]:+.2f} cm/s', 760, 639, .37)
    text('Slip: simulator evaluation only', 760, 658, .35, (150, 165, 180))
    text('W/S forward/back | A/D turn | SPACE stop | M auto/manual | R reset | P pause', 270, 680, 0.45)
    text('[ / ] browse paths | 0 follow latest | P pause to inspect | ESC close', 270, 707, 0.45)
    text('POSITION ERROR | EKF estimate compared with actual robot position', 30, 748, .57)
    samples = evaluation.samples
    if samples:
        last_time = max(10., samples[-1]['time'])
        left, right, top, bottom = 70, 1060, 785, 900
        maximum = max(.01, max(s['estimation_error_m'] for s in samples)*1.15)

        def graph_point(t, value):
            return (int(left+t/last_time*(right-left)), int(bottom-value/maximum*(bottom-top)))

        cv2.rectangle(canvas, (left, top), (right, bottom), (75, 85, 95), 1)
        for tick in [0, maximum/2, maximum]:
            y = graph_point(0, tick)[1]
            cv2.line(canvas, (left, y), (right, y), (45, 55, 65), 1)
            text(f'{tick*100:.1f}', left-45, y+5, .4)
        points = np.array([graph_point(s['time'], s['estimation_error_m']) for s in samples], np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, (235, 220, 70), 2)
        else:
            cv2.circle(canvas, tuple(points[0]), 2, (235, 220, 70), -1)
        text('cm', left-45, 770, .4)
        for t in np.linspace(0, last_time, 6):
            x = graph_point(t, 0)[0]
            cv2.line(canvas, (x, bottom), (x, bottom+4), grey, 1)
            text(f'{t:.0f}s', x-10, 922, .4)
        text(f'AVERAGE POSITION ERROR: {average_text}', 30, 955, .65, (235, 220, 70))
        text('Mean distance between estimated and actual XY position; automatic run through docking.', 30, 986, .46)
    return canvas
