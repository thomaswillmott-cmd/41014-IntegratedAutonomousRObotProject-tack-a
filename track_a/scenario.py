"""Reproducible experiments; geometry is not passed into sensor-map navigation."""
from dataclasses import dataclass, field
import numpy as np

GOAL = np.array([2.30, 1.4, 0.0])
STAGING = np.array([1.65, 1.4])


@dataclass
class Scenario:
    seed: int = 1
    layout: str = 'fixed'
    objective: str = 'time'
    mapping: str = 'lidar'
    controller: str = 'lqr'
    clean: bool = False  # Disable injected errors; keep contact physics and image detection.
    noisy: bool = False
    dropout: bool = False
    disturbance: bool = False
    blocked: bool = False
    recovery: bool = True
    wheel_feedback: bool = True
    camera_calibration: bool = True
    q_scale: float = 1.
    start: np.ndarray = field(default_factory=lambda: np.array([-2., -1.5, 0.]))

    def __post_init__(self):
        if self.clean and (self.noisy or self.dropout or self.disturbance):
            raise ValueError("Clean mode cannot be combined with injected-error options")

    def obstacles(self):
        if self.layout == 'fixed':
            return [('crate', 'box', (-.25, .35), (.35, .35)),
                    ('crate_2', 'box', (1., -1.1), (.30, .45)),
                    ('barrel', 'cylinder', (.7, .2), (.25, .25))]
        rng = np.random.default_rng(self.seed)
        obstacles = []
        for attempt in range(500):
            if len(obstacles) == 5:
                break
            centre = rng.uniform([-1.1, -1.5], [1.25, 1.35])
            radius = float(rng.uniform(.18, .30))
            if np.linalg.norm(centre-self.start[:2]) < .8 or np.linalg.norm(centre-STAGING) < .8:
                continue
            if any(np.linalg.norm(centre-np.array(o[2])) < radius+max(o[3])+.50 for o in obstacles):
                continue
            obstacles.append((f'obstacle_{len(obstacles)}', 'cylinder', tuple(centre), (radius, radius)))
        return obstacles

    def as_dict(self):
        return {**self.__dict__, 'start': self.start.tolist(), 'obstacles': self.obstacles()}
