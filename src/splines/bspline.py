import numpy as np
import casadi as cas

from .basis import bspline_basis


class BSpline:
    def __init__(self, control_points):
        self.control_points = control_points  # (N, dim)
        self.n_control_points = self.control_points.shape[0]

        self.upper_bound = self.n_control_points - 3

    def _basis_mat(self, ts, der=0):
        mat = np.zeros((ts.shape[0], self.n_control_points))
        for i, t in enumerate(ts):
            offset = max(min(int(np.floor(t)), self.upper_bound - 1), 0)
            u = t - offset

            mat[i][offset : offset + 4] = bspline_basis(u, der=der)

        return mat

    def spline_eval(self, N: int, der: int = 0) -> np.ndarray:
        ts = np.linspace(0, self.upper_bound, N)
        basis = self._basis_mat(ts, der=der)

        return basis @ self.control_points
