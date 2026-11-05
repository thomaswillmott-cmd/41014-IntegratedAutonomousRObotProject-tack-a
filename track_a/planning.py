"""LiDAR occupancy mapping and heading-aware A*; no simulator access.

Unknown cells are provisionally traversable. Every plan is checked again as new
scans arrive. Robot footprint inflation supplies collision clearance.
"""
import heapq
import math
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .scenario import STAGING

RESOLUTION = .02
ORIGIN = np.array([-3.04, -2.56])
MAP_MAX = np.array([3.04, 2.56])
GRID_SHAPE = tuple((np.rint((MAP_MAX-ORIGIN)/RESOLUTION).astype(int)+1)[::-1])
DIRECTIONS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])


class OccupancyMap:
    def __init__(self, known_obstacles=None):
        self.logodds = np.zeros(GRID_SHAPE, np.float32)
        self.known = np.zeros_like(self.logodds, dtype=bool)
        self.observed = np.zeros_like(self.logodds, dtype=bool)
        self.known[[0, -1], :] = True
        self.known[:, [0, -1]] = True
        if known_obstacles is not None:
            yy, xx = np.indices(self.logodds.shape)
            points = ORIGIN + np.stack([xx, yy], axis=-1)*RESOLUTION
            for _, kind, centre, size in known_obstacles:
                delta = points-np.array(centre)
                mask = np.linalg.norm(delta, axis=-1) <= size[0] if kind == 'cylinder' else np.all(np.abs(delta) <= size, axis=-1)
                self.known |= mask
        self.blocked = self.known.copy()

    def cell(self, xy):
        ij = np.rint((np.asarray(xy)-ORIGIN)/RESOLUTION).astype(int)
        return int(ij[0]), int(ij[1])

    def xy(self, cell):
        return ORIGIN + np.array(cell)*RESOLUTION

    def inside(self, cell):
        x, y = cell
        return 0 <= x < self.logodds.shape[1] and 0 <= y < self.logodds.shape[0]

    def update(self, pose, angles, ranges):
        origin = pose[:2] + np.array([np.cos(pose[2]), np.sin(pose[2])])*(-.049)
        start = self.cell(origin)
        free = np.zeros_like(self.logodds, np.uint8)
        hits = np.zeros_like(free)
        for angle, distance in zip(angles, ranges):
            heading = pose[2]+angle
            end = self.cell(origin+distance*np.array([np.cos(heading), np.sin(heading)]))
            cv2.line(free, start, end, 1, 1)
            if distance < 4.95 and self.inside(end):
                hits[end[1], end[0]] = 1
        self.observed |= free.astype(bool)
        self.logodds[free.astype(bool)] -= .25
        self.logodds[hits.astype(bool)] += 1.2
        np.clip(self.logodds, -3, 4, out=self.logodds)

    def inflate(self, margin=.33):
        occupied = (self.logodds > .5) | self.known
        self.blocked = distance_transform_edt(~occupied)*RESOLUTION < margin
        return self.blocked

    def clear_segment(self, a, b):
        length = np.linalg.norm(np.asarray(b)-a)
        for fraction in np.linspace(0, 1, max(2, int(length/(RESOLUTION*.4))+1)):
            cell = self.cell(np.asarray(a)*(1-fraction)+np.asarray(b)*fraction)
            if not self.inside(cell) or self.blocked[cell[1], cell[0]]:
                return False
        return True


def astar(grid, start_xy, goal_xy=STAGING, heading=0., objective='time'):
    """Optimal for this discrete 8-heading graph and specified nonnegative cost.

    time cost = distance/0.18 + turn/0.9, an approximate stop/turn travel time.
    Not a claim of globally minimum-time continuous robot dynamics.
    """
    start, goal = grid.cell(start_xy), grid.cell(goal_xy)
    if not grid.inside(start) or not grid.inside(goal):
        return None
    if grid.blocked[goal[1], goal[0]]:
        return None
    initial = (*start, int(round(heading/(np.pi/4))) % 8)
    scale = 1/.18 if objective == 'time' else 1.
    def heuristic(x, y):
        return math.hypot(x-goal[0], y-goal[1])*RESOLUTION*scale
    queue = [(heuristic(*start), 0., initial)]
    costs, previous = {initial: 0.}, {}
    last = None
    while queue:
        _, g, node = heapq.heappop(queue)
        if g > costs[node]+1e-9:
            continue
        x, y, old_direction = node
        if (x, y) == goal:
            last = node
            break
        for direction, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x+int(dx), y+int(dy)
            if not grid.inside((nx, ny)) or grid.blocked[ny, nx]:
                continue
            if dx and dy and (grid.blocked[y, nx] or grid.blocked[ny, x]):
                continue
            turn = min((direction-old_direction) % 8, (old_direction-direction) % 8)*np.pi/4
            step_cost = math.hypot(dx, dy)*RESOLUTION*scale
            if objective == 'time':
                step_cost += turn/.9
            child = (nx, ny, direction)
            candidate = g+step_cost
            if candidate < costs.get(child, float('inf'))-1e-9:
                costs[child] = candidate
                previous[child] = node
                heapq.heappush(queue, (candidate+heuristic(nx, ny), candidate, child))
    if last is None:
        return None
    cells = [last]
    while cells[-1] != initial:
        cells.append(previous[cells[-1]])
    points = [np.asarray(start_xy)] + [grid.xy(c[:2]) for c in cells[::-1][1:]] + [np.asarray(goal_xy)]
    # Collision-checked shortcutting produces a shorter continuous polyline.
    result = [points[0]]
    i = 0
    while i < len(points)-1:
        j = len(points)-1
        while j > i+1 and not grid.clear_segment(points[i], points[j]):
            j -= 1
        result.append(points[j])
        i = j
    return np.array(result)
