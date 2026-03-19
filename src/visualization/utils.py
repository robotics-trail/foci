import numpy as np
import casadi as cas
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass(frozen=True)
class CasadiFKMidpoints:
    robot: object
    n_links: int

    def midpoints(self, curve: np.ndarray) -> np.ndarray:
        num_samples = curve.shape[0]
        out = np.zeros((num_samples, self.n_links, 3), dtype=float)

        for i in range(num_samples):
            q = curve[i]

            fk_flat = (
                np.array(self.robot.forward_kinematics(q)).astype(float).reshape(-1)
            )
            joint_positions = fk_flat.reshape(self.n_links + 1, 3)

            for j in range(self.n_links):
                out[i, j] = 0.5 * (joint_positions[j] + joint_positions[j + 1])

        return out


@dataclass
class CasadiFKGaussians:
    robot: object
    n_links: int
    fk_fun: cas.Function = field(init=False)

    def __post_init__(self):
        q_sym = cas.MX.sym("q", self.robot.get_n_joints())
        self.fk_fun = cas.Function(
            "fk_gaussians_utils",
            [q_sym],
            [self.robot.forward_kinematics(q_sym)],
        )

    def gaussian_points(
        self,
        curve: np.ndarray,
        gaussian_specs: List[Tuple[int, float]],
    ) -> np.ndarray:
        curve = np.asarray(curve, dtype=float)
        num_samples = curve.shape[0]
        n_total_gaussians = len(gaussian_specs)

        if n_total_gaussians == 0:
            return np.zeros((num_samples, 0, 3), dtype=float)

        points = np.zeros((num_samples, n_total_gaussians, 3), dtype=float)

        for i in range(num_samples):
            fk_flat = np.array(self.fk_fun(curve[i])).astype(float).reshape(-1)
            joint_positions = fk_flat.reshape(self.n_links + 1, 3)

            for g_idx, (link_idx, t) in enumerate(gaussian_specs):
                if not (0 <= link_idx < self.n_links):
                    raise ValueError(
                        f"Invalid link_idx={link_idx}. Expected 0 <= link_idx < {self.n_links}."
                    )
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        f"Invalid interpolation factor t={t}. Expected 0.0 <= t <= 1.0."
                    )

                p0 = joint_positions[link_idx]
                p1 = joint_positions[link_idx + 1]
                points[i, g_idx] = (1.0 - t) * p0 + t * p1

        return points


class GaussianModel:
    def cov(self, link_idx: int) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class SharedCovariance(GaussianModel):
    cov_mat: np.ndarray  # (3,3)

    def cov(self, link_idx: int) -> np.ndarray:
        return self.cov_mat


@dataclass(frozen=True)
class PerLinkCovariances(GaussianModel):
    covs: np.ndarray  # (n_links,3,3)

    def cov(self, link_idx: int) -> np.ndarray:
        return self.covs[link_idx]


@dataclass(frozen=True)
class EllipsoidFactory:
    n_std: float = 2.0

    def cov_to_ellipsoid(self, cov: np.ndarray):
        eigvals, eigvecs = np.linalg.eigh(cov)

        radii = self.n_std * np.sqrt(np.maximum(eigvals, 1e-12))

        Rm = eigvecs
        if np.linalg.det(Rm) < 0:
            Rm[:, 0] *= -1

        quat_xyzw = R.from_matrix(Rm).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return radii, quat_wxyz
