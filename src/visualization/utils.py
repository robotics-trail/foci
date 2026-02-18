import numpy as np
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass


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
