import numpy as np
import casadi as cas

from .basis import BSPLINE_3, BSPLINE_2, B_SPLINE_1, B_SPLINE_0
from .basis import MINVO_3, MINVO_2, MINVO_1, MINVO_0


def _diff(V):
    V_type = type(V).__name__

    # Compatible with casadi
    if V_type == "MX":
        result = cas.MX.zeros(V.shape[0] - 1, V.shape[1])

        for i in range(V.shape[0] - 1):
            result[i] = V[i + 1] - V[i]

        return result

    # Compatible with numpy
    else:
        return V[1:] - V[:-1]


def minvo_hulls(control_points, derivative=0):
    assert derivative in {0, 1, 2, 3}

    V = control_points

    d1 = _diff(V)
    d2 = _diff(d1) if derivative >= 2 else None
    d3 = _diff(d2) if derivative >= 3 else None

    hulls = []

    if derivative == 0:
        for i in range(V.shape[0] - 3):
            hull = np.linalg.inv(MINVO_3) @ BSPLINE_3 @ V[i : i + 4, :]
            hulls.append(hull)

    elif derivative == 1:
        for i in range(V.shape[0] - 4):
            hull = np.linalg.inv(MINVO_2) @ BSPLINE_2 @ d1[i : i + 3, :]
            hulls.append(hull)

    elif derivative == 2:
        for i in range(V.shape[0] - 5):
            hull = np.linalg.inv(MINVO_1) @ B_SPLINE_1 @ d2[i : i + 2, :]
            hulls.append(hull)

    elif derivative == 3:
        for i in range(V.shape[0] - 6):
            hull = np.linalg.inv(MINVO_0) @ B_SPLINE_0 @ d3[i : i + 1, :]
            hulls.append(hull)

    return hulls
