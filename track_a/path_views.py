"""Saved path selection and graph traces."""
import colorsys
import math


def path_rgb(path_id):
    return colorsys.hsv_to_rgb((.08+(int(path_id)-1)*.38196601125) % 1., .75, .88)


def path_bgr(path_id):
    return tuple(int(round(channel*255)) for channel in path_rgb(path_id)[::-1])


def path_runs(samples, key='tracking_error_m'):
    """Start a new trace at each path change or missing sample."""
    result = []
    current = None
    for sample in samples:
        value = sample[key]
        if value is None or not math.isfinite(float(value)):
            current = None
            continue
        path_id = int(sample['path_id'])
        if current is None or current['path_id'] != path_id:
            current = dict(path_id=path_id, time=[], value=[])
            result.append(current)
        current['time'].append(float(sample['time']))
        current['value'].append(float(value))
    return result


def selected_path(history, requested=None):
    """None selects the latest reference."""
    ids = [version['path_id'] for version in history]
    return requested if requested in ids else (ids[-1] if ids else None)


def step_path(history, requested, direction):
    """Browse paths for display only."""
    ids = [version['path_id'] for version in history]
    if not ids:
        return None
    index = ids.index(selected_path(history, requested))
    return ids[max(0, min(len(ids)-1, index+direction))]
