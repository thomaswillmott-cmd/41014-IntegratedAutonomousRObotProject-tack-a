"""State-space model and Jacobians for the analysis checks."""
import numpy as np
from .world import CAMERA_OFFSET, MARKER_XY


def state_derivative(state, control):
    v, omega = control
    return np.array([v*np.cos(state[2]), v*np.sin(state[2]), omega])


def linearisation(heading, speed, dt=.02):
    A = np.eye(3)
    A[0, 2] = -speed*np.sin(heading)*dt
    A[1, 2] = speed*np.cos(heading)*dt
    B = dt*np.array([[np.cos(heading), 0], [np.sin(heading), 0], [0, 1]])
    return A, B


def measurement_jacobian(state, include_yaw=True):
    theta = state[2]
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = MARKER_XY-state[:2]-CAMERA_OFFSET*np.array([c, s])
    q = dx*dx+dy*dy
    r = np.sqrt(q)
    H = np.array([[-dx/r, -dy/r, CAMERA_OFFSET*(dx*s-dy*c)/r],
                  [dy/q, -dx/q, -CAMERA_OFFSET*(dx*c+dy*s)/q-1]])
    return np.vstack([H, [0, 0, 1]]) if include_yaw else H


def controllability(A, B):
    return np.hstack([B, A@B, A@A@B])


def observability(A, H):
    return np.vstack([H, H@A, H@A@A])
