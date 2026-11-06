"""Graphs from saved evaluation data only."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .evaluation import nearest_on_path
from .scenario import GOAL, STAGING


DOCK_PHASES = {'ACQUIRE', 'DOCK', 'VERIFY', 'DOCKED'}


def initial_plan(history):
    """Keep the first usable navigation path unchanged."""
    for version in history:
        if version['kind'] == 'navigation' and len(version['points']) >= 2:
            return version
    return None


def reference_deviations(samples, version):
    """Missing references stay NaN; docking is excluded from initial-plan deviation."""
    values = np.full(len(samples), np.nan)
    if version is None:
        return values
    for i, sample in enumerate(samples):
        if (sample['scored'].lower() in ('true', '1') and
                float(sample['time']) >= version['time'] and sample['phase'] not in DOCK_PHASES):
            _, values[i] = nearest_on_path([float(sample['true_x']), float(sample['true_y'])], version['points'])
    return values


def rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if len(values) else None


def trial_report(folder):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    history = json.loads((folder/'path_history.json').read_text(encoding='utf-8'))
    with (folder/'position_errors.csv').open(newline='') as stream:
        samples = list(csv.DictReader(stream))
    data = np.atleast_1d(np.genfromtxt(folder/'trial.csv', delimiter=',', names=True))
    version = initial_plan(history)
    fixed = reference_deviations(samples, version)
    times = np.array([float(s['time']) for s in samples])
    active = np.array([float(s['tracking_error_m']) if s['tracking_error_m'] and s['scored'].lower() in ('true', '1')
                       else np.nan for s in samples])
    docking = np.array([s['phase'] in DOCK_PHASES for s in samples], dtype=bool)
    with (folder/'reference_errors.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'initial_plan_deviation_m', 'active_path_tracking_error_m'])
        writer.writerows((t, f if np.isfinite(f) else '', a if np.isfinite(a) else '')
                         for t, f, a in zip(times, fixed, active))
    metrics = dict(seed=summary['seed'], docking_success=summary['docking_success'],
                   simulated_seconds=summary['simulated_seconds'], docking_time_s=summary['docking_time_s'],
                   collision_control_steps=summary['collision_control_steps'],
                   average_position_error_m=summary['average_position_error_m'],
                   position_rmse_m=summary['position_rmse_m'],
                   active_path_tracking_rmse_m=rms(active), initial_plan_deviation_rmse_m=rms(fixed),
                   navigation_tracking_rmse_m=rms(active[~docking]), docking_tracking_rmse_m=rms(active[docking]),
                   final_docking_distance_m=summary['final_true_goal_distance_m'],
                   final_heading_error_deg=float(np.rad2deg(summary['final_heading_error_rad'])),
                   control_effort=summary['control_effort'],
                   initial_path_id=version['path_id'] if version else None,
                   initial_plan_note='First usable navigation path, unchanged. Deviation scored before docking phases only; replanning is not a control failure.',
                   source_sha256=summary['source_sha256'])
    (folder/'assessment_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for name, kind, centre, size in summary['scenario']['obstacles']:
        if kind == 'cylinder':
            ax.add_patch(Circle(centre, size[0], color='grey', alpha=.35))
        else:
            ax.add_patch(Rectangle(np.array(centre)-size, 2*size[0], 2*size[1], color='grey', alpha=.35))
    if version:
        points = np.array(version['points'])
        ax.plot(*points.T, 'o--', color='#b87909', label=f"Initial plan (path {version['path_id']})")
    ax.plot(*np.array([STAGING, GOAL[:2]]).T, ':', color='green', label='Separate docking reference')
    ax.plot(data['true_x'], data['true_y'], color='#187eae', label='Actual trajectory (evaluation truth)')
    ax.plot(data['est_x'], data['est_y'], '--', color='#a63d8e', alpha=.7, label='EKF estimate')
    ax.plot(*GOAL[:2], '*', color='green', markersize=12, label='Dock target')
    ax.set(xlabel='World x (m)', ylabel='World y (m)', title=f"Seed {summary['seed']}: actual trajectory versus first planned path")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(folder/'trajectory_reference.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(times, fixed*100, color='#b87909')
    axes[0].set(title='Deviation from the fixed initial plan (navigation only)', ylabel='Distance (cm)')
    axes[1].plot(times, active*100, color='#187eae')
    if np.any(docking):
        axes[1].axvline(times[docking][0], color='grey', linestyle='--', label='Docking reference begins')
        axes[1].legend(fontsize=8)
    axes[1].set(title='Tracking error to the active reference (navigation and docking)', xlabel='Time (s)', ylabel='Distance (cm)')
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    fig.suptitle('Nearest finite-polyline distance; initial-plan deviation includes deliberate replanning')
    fig.savefig(folder/'tracking_errors.png', dpi=160)
    plt.close(fig)

    distance = np.hypot(data['true_x']-GOAL[0], data['true_y']-GOAL[1])
    heading = np.rad2deg(np.arctan2(np.sin(data['true_yaw']-GOAL[2]), np.cos(data['true_yaw']-GOAL[2])))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(data['time'], distance*100)
    axes[0].axhline(8, linestyle='--', color='grey', label='Offline success limit: <8 cm')
    axes[0].set(ylabel='Distance to dock (cm)', title=f"Final distance: {distance[-1]*100:.2f} cm; success: {summary['docking_success']}")
    axes[0].legend(fontsize=8)
    axes[1].plot(data['time'], heading)
    for limit in [-np.rad2deg(.12), np.rad2deg(.12)]:
        axes[1].axhline(limit, linestyle='--', color='grey')
    axes[1].set(ylabel='Signed heading error (degrees)', xlabel='Time (s)', title=f"Final absolute heading error: {metrics['final_heading_error_deg']:.3f} degrees")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(folder/'docking_errors.png', dpi=160)
    plt.close(fig)
    from plot_results import plot_position_errors
    plot_position_errors(folder)
    return metrics
