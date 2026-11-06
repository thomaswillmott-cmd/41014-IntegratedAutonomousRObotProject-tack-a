"""Three layouts back to back, with all errors and saved results."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import secrets
import subprocess
import sys

import numpy as np

from track_a.navigation import lqr_gain
from track_a.trial_report import trial_report
from track_a.world import ROOT


def controller_notes():
    K, A, B, Q, R = lqr_gain(.18)
    eigenvalues = np.linalg.eigvals(A-B@K)
    return ('## Controller design and justification\n\n'
            'Navigation uses scheduled discrete LQR on estimated along-track, lateral and heading errors, '
            'at a 0.02 s control period. Q=diag(6,14,3), R=diag(2,1). '
            'The lateral weight is larger to discourage cross-track departures; heading feedback aligns '
            'the vehicle, and input penalties moderate speed and turn corrections. These are engineering '
            'choices, with different units; this batch does not prove that they are optimal.\n\n'
            f'At nominal speed 0.18 m/s, K={np.round(K, 5).tolist()}; '
            f'local closed-loop eigenvalues={np.round(eigenvalues, 6).tolist()}. '
            'This is a local linear result. Saturation, obstacle stops, wheel feedback and replanning '
            'are outside that proof. Final docking uses separate camera-gated pose feedback. '
            'Docking uses v=direction*min(0.10,0.65*distance)*max(0,cos(alpha)) and '
            'omega=clip(2*alpha-0.65*beta,-0.65,0.65), where alpha is the target-bearing '
            'error and beta is the final-orientation term. The distance gain slows the '
            'approach near the goal; the angular terms balance aiming at the goal with '
            'aligning at the dock. The angular verifier uses gain 2, limited to 0.4 rad/s, '
            'and holds for over 0.6s once estimated heading error is below 0.08rad. '
            'These gains are engineering settings, not an empirically optimal selection. '
            'See track_a/navigation.py for the full hybrid law. '
            'Reported tracking error, docking error, completion time and control effort support evaluation '
            'of these settings; a matched gain/controller comparison is needed to establish a tuning benefit.\n')


def save_report(output, seeds, results, duration):
    (output/'all_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    fields = ['trial', 'seed', 'docking_success', 'docking_time_s', 'simulated_seconds',
              'collision_control_steps', 'average_position_error_m', 'position_rmse_m',
              'initial_plan_deviation_rmse_m', 'active_path_tracking_rmse_m',
              'final_docking_distance_m', 'final_heading_error_deg', 'control_effort', 'error']
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    successes = sum(bool(r.get('docking_success')) for r in results)
    lines = ['# Sequential random-layout assessment', '',
             f'Scheduled seeds: {seeds}. Maximum {duration:g} simulated seconds each. All injected errors enabled.',
             f'Completed attempts: {len(results)}/{len(seeds)}. Docking successes: {successes}/{len(results)} attempted.', '',
             'All attempts, timeouts and errors are retained. Random seeds are saved before running so the batch can be repeated.', '',
             '| Trial | Seed | Docked | Duration (s) | Mean estimation error (cm) | Active tracking RMSE (cm) | Final dock distance (cm) | Final heading error (deg) | Contact steps |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    def number(r, key, scale=1):
        value = r.get(key)
        return f'{value*scale:.2f}' if value is not None else '-'
    for r in results:
        lines.append(f"| {r['trial']} | {r['seed']} | {r.get('docking_success', False)} | {number(r, 'simulated_seconds')} | {number(r, 'average_position_error_m', 100)} | {number(r, 'active_path_tracking_rmse_m', 100)} | {number(r, 'final_docking_distance_m', 100)} | {number(r, 'final_heading_error_deg')} | {r.get('collision_control_steps', '-')} |")
        if r.get('error'):
            lines.append(f"\nTrial {r['trial']} error: {r['error']}\n")
    lines += ['', '## Interpreting the plots', '',
              '- trajectory_reference.png: actual trajectory, EKF estimate, unchanged first usable path, and separate dock approach.',
              '- tracking_errors.png: initial-plan deviation and active-path tracking error in separate panels. '
              'The first plan can cross undiscovered obstacles. Deliberate replanning can increase initial-plan deviation without indicating poor control. '
              'The initial plan ends at staging, so its deviation excludes docking phases.',
              '- docking_errors.png: true distance and heading relative to the dock target, with offline success limits.',
              '- position_errors.png: EKF estimation error, distinct from tracking error.',
              '- comparison.csv and each trial/assessment_metrics.json: numeric results. Distances in CSV are metres; headings are degrees.', '',
              'Tracking uses nearest distance to a finite polyline, not time-synchronised trajectory error. '
              'Active-reference switches can cause discontinuities; path_history.json preserves their provenance. '
              'Ground truth is evaluation-only and never fed to the controller. '
              'The control-effort value is the integral of commanded v squared plus omega squared, '
              'a mixed-unit comparison proxy rather than energy consumption. '
              'Three trials are preliminary evidence and do not establish safety for arbitrary layouts. '
              'Success additionally requires the controller to finish, final true distance <0.08m, '
              'heading error <0.12rad and no obstacle contacts.', '',
              'Errors: encoder/LiDAR/camera noise, unequal wheel drive, forced 12–18s camera blackout '
              'plus random frame losses, and the sideways push at 10s. The optional newly inserted obstacle is not enabled.', '',
              controller_notes()]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Run faster without windows; camera sensing remains enabled')
    parser.add_argument('--seeds', type=int, nargs=3, help='Repeat a batch with exactly these three seeds')
    parser.add_argument('--duration', type=float, default=180)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be finite and positive')
    seeds = args.seeds
    if seeds is None:
        seeds = []
        while len(seeds) < 3:
            candidate = secrets.randbelow(1_000_000)
            if candidate not in seeds:
                seeds.append(candidate)
    if any(seed < 0 for seed in seeds):
        parser.error('Seeds must be nonnegative')
    output = args.output or ROOT/'results'/('batch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(seeds=seeds, duration=args.duration, all_errors=True, layout='random', headless=args.headless)
    (output/'batch_config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Batch seeds: {seeds}\nResults: {output}', flush=True)
    results = []
    for index, seed in enumerate(seeds, 1):
        folder = output/f'trial_{index}_seed_{seed}'
        command = [sys.executable, str(ROOT/'run_robot.py'), '--exit-on-complete', '--layout', 'random',
                   '--seed', str(seed), '--all-errors', '--duration', str(args.duration), '--output', str(folder)]
        if args.headless:
            command.append('--headless')
        print(f'\nTrial {index}/3 — seed {seed}', flush=True)
        try:
            with (output/f'trial_{index}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = trial_report(folder)
            metrics['trial'] = index
            results.append(metrics)
            print(f"Docked: {metrics['docking_success']} | time: {metrics['simulated_seconds']:.1f}s | "
                  f"mean position error: {100*metrics['average_position_error_m']:.2f}cm | contacts: {metrics['collision_control_steps']}", flush=True)
        except Exception as error:
            results.append(dict(trial=index, seed=seed, docking_success=False, error=str(error)))
            print(f'Trial failed: {error}; see trial_{index}.log', flush=True)
        save_report(output, seeds, results, args.duration)
    print(f'\nBatch complete. Open {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
