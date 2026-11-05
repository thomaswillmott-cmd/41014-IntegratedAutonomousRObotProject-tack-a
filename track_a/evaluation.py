"""Evaluation-only active-path history and continuous position error.

This module never supplies pose or reference corrections to navigation/control.
"""
import csv
import json
from pathlib import Path
import numpy as np

from .scenario import STAGING, GOAL
from .world import WAYPOINTS


def nearest_on_path(position, path):
    """Euclidean distance to finite polyline segments (endpoints included)."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    position = np.asarray(position, dtype=float)[:2]
    if len(path) == 0:
        return None, None
    if len(path) == 1:
        return path[0].copy(), float(np.linalg.norm(position-path[0]))
    a, b = path[:-1], path[1:]
    delta = b-a
    squared = np.sum(delta*delta, axis=1)
    fraction = np.sum((position-a)*delta, axis=1)/np.maximum(squared, 1e-12)
    projection = a + np.clip(fraction, 0, 1)[:, None]*delta
    distances = np.linalg.norm(projection-position, axis=1)
    index = int(np.argmin(distances))
    return projection[index].copy(), float(distances[index])


class PathEvaluation:
    def __init__(self):
        self.history = []
        self.samples = []
        self.path = np.empty((0, 2))
        self.kind = None
        self.path_id = 0
        self.tracking_squared = 0.
        self.tracking_count = 0
        self.estimation_squared = 0.
        self.estimation_total = 0.
        self.scored_count = 0
        self.current = None

    def activate(self, path, kind, now, estimate, truth, reason):
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        unchanged = kind == self.kind and path.shape == self.path.shape and np.allclose(path, self.path, atol=1e-9, rtol=0)
        if unchanged:
            return
        self.path = path.copy()
        self.kind = kind
        self.path_id += 1
        self.history.append(dict(path_id=self.path_id, time=float(now), kind=kind, reason=reason,
                                 points=path.tolist(), estimated_checkpoint=np.asarray(estimate).tolist(),
                                 true_checkpoint=np.asarray(truth).tolist()))

    def observe_reference(self, driver, now, estimate, truth):
        phase = getattr(driver, 'phase', 'BASELINE')
        if phase in ('ACQUIRE', 'DOCK', 'VERIFY', 'DOCKED'):
            path = np.array([STAGING, GOAL[:2]])
            kind = 'dock_approach'
            reason = 'Transition to final docking reference'
        elif hasattr(driver, 'path'):
            path = driver.path
            kind = 'navigation'
            reason = 'Initial plan' if not self.history else 'Planner updated active route'
        else:
            path = np.vstack([self.history[0]['estimated_checkpoint'][:2] if self.history else estimate[:2], WAYPOINTS])
            kind = 'baseline'
            reason = 'Fixed baseline reference'
        self.activate(path, kind, now, estimate, truth, reason)

    def measure(self, now, estimate, truth, phase, scored=True):
        reference, tracking = nearest_on_path(truth[:2], self.path)
        _, estimated_tracking = nearest_on_path(estimate[:2], self.path)
        estimation = float(np.linalg.norm(np.asarray(estimate)[:2]-np.asarray(truth)[:2]))
        return dict(time=float(now), path_id=self.path_id, phase=phase,
                    true_x=float(truth[0]), true_y=float(truth[1]),
                    est_x=float(estimate[0]), est_y=float(estimate[1]),
                    reference_x=float(reference[0]) if reference is not None else None,
                    reference_y=float(reference[1]) if reference is not None else None,
                    tracking_error_m=tracking, estimated_tracking_error_m=estimated_tracking,
                    estimation_error_m=estimation, scored=bool(scored))

    def sample(self, now, estimate, truth, phase, scored=True):
        record = self.measure(now, estimate, truth, phase, scored)
        if scored:
            self.scored_count += 1
            self.estimation_squared += record['estimation_error_m']**2
            self.estimation_total += record['estimation_error_m']
            if record['tracking_error_m'] is not None:
                self.tracking_squared += record['tracking_error_m']**2
                self.tracking_count += 1
        record['cumulative_tracking_rmse_m'] = self.tracking_rmse
        record['cumulative_estimation_rmse_m'] = self.estimation_rmse
        record['average_position_error_m'] = self.average_position_error
        self.current = record
        self.samples.append(record)

    @property
    def tracking_rmse(self):
        return float(np.sqrt(self.tracking_squared/self.tracking_count)) if self.tracking_count else None

    @property
    def average_position_error(self):
        """Arithmetic mean of XY error magnitudes over scored 10-Hz samples."""
        return self.estimation_total/self.scored_count if self.scored_count else None

    @property
    def estimation_rmse(self):
        return float(np.sqrt(self.estimation_squared/self.scored_count)) if self.scored_count else None

    def save(self, folder):
        folder = Path(folder)
        (folder/'path_history.json').write_text(json.dumps(self.history, indent=2), encoding='utf-8')
        if self.samples:
            with (folder/'position_errors.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.samples[0]))
                writer.writeheader()
                writer.writerows(self.samples)
