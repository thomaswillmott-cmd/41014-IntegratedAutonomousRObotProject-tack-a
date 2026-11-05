"""Plot a saved trial: python plot_results.py results/validation_seed1"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from track_a.world import ROOT, WAYPOINTS
import json


def plot_position_errors(folder):
    """Single position-estimation chart; detailed tracking data stays in the CSV."""
    folder = Path(folder)
    if not (folder/'position_errors.csv').exists():
        return
    data = np.atleast_1d(np.genfromtxt(folder/'position_errors.csv', delimiter=',', names=True, dtype=None, encoding='utf-8'))
    time_axis = np.asarray(data['time'], dtype=float)
    error_cm = np.asarray(data['estimation_error_m'], dtype=float)*100
    scored = np.isin(np.asarray(data['scored']).astype(str), ['True', 'true', '1'])
    average_cm = float(np.mean(error_cm[scored])) if np.any(scored) else None
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(time_axis, error_cm, color='#187eae', linewidth=1.8, label='Position error')
    if average_cm is not None:
        ax.axhline(average_cm, color='#a76d08', linestyle='--', linewidth=1.2,
                   label=f'Average: {average_cm:.2f} cm')
    if np.any(scored) and time_axis[-1] > time_axis[scored][-1]+.2:
        ax.axvspan(time_axis[scored][-1], time_axis[-1], color='grey', alpha=.08,
                   label='After scoring ended')
    ax.set(xlabel='Time (s)', ylabel='Position error (cm)',
           title='Position error | EKF estimate compared with actual robot position', ylim=(0, None))
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(folder/'position_errors.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=ROOT/'results'/'latest')
    args = parser.parse_args()
    data = np.genfromtxt(args.folder/'trial.csv', delimiter=',', names=True)
    data = np.atleast_1d(data)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(data['true_x'], data['true_y'], label='Simulator truth (evaluation)')
    ax.plot(data['est_x'], data['est_y'], '--', label='EKF estimate')
    summary_path = args.folder/'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    if 'scenario' not in summary or summary.get('scenario', {}).get('controller') == 'baseline':
        ax.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1], 'o:', color='grey', label='Baseline waypoints')
    else:
        from track_a.scenario import STAGING, GOAL
        ax.plot(*STAGING, 'o', color='grey', label='Dock staging')
        ax.plot(*GOAL[:2], '*', color='green', label='Dock goal')
    ax.set(xlabel='World x (m)', ylabel='World y (m)', title='Route')
    ax.axis('equal')
    ax.legend(fontsize=8)
    error = np.hypot(data['true_x']-data['est_x'], data['true_y']-data['est_y'])
    axes[0, 1].plot(data['time'], error)
    axes[0, 1].set(xlabel='Time (s)', ylabel='Position error (m)', title='EKF error against simulator truth')
    for state, variance, label in [('x', 'var_x', 'x'), ('y', 'var_y', 'y')]:
        error = data[f'est_{state}']-data[f'true_{state}']
        line, = axes[1, 0].plot(data['time'], error, label=f'{label} error')
        sigma = 2*np.sqrt(data[variance])
        axes[1, 0].fill_between(data['time'], -sigma, sigma, alpha=0.10, color=line.get_color())
    axes[1, 0].set(xlabel='Time (s)', ylabel='Signed error (m)', title='Position errors and marginal +/-2 sigma')
    axes[1, 0].legend()
    axes[1, 1].step(data['time'], data['camera_updates'], label='Accepted image updates')
    axes[1, 1].set(xlabel='Time (s)', ylabel='Cumulative updates', title='Marker observations reaching EKF')
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    path = args.folder/'performance.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_position_errors(args.folder)
    print(f'Plot saved: {path}')


if __name__ == '__main__':
    main()
