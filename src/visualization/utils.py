"""
Helper utilities for visualization-related kinematics and Gaussian geometry.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import casadi as cas
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class CasadiFKMidpoints:
    """
    Compute link midpoints along a trajectory using the robot forward kinematics.

    Parameters
    ----------
    robot : object
        Robot object exposing `forward_kinematics(q)`.
    n_links : int
        Number of links in the active chain.
    """

    robot: object
    n_links: int

    def midpoints(self, curve: np.ndarray) -> np.ndarray:
        """
        Compute midpoint positions for every link and trajectory sample.

        Parameters
        ----------
        curve : np.ndarray
            Joint trajectory of shape `(num_samples, n_joints)`.

        Returns
        -------
        np.ndarray
            Midpoints of shape `(num_samples, n_links, 3)`.
        """
        num_samples = curve.shape[0]
        out = np.zeros((num_samples, self.n_links, 3), dtype=float)

        for sample_idx in range(num_samples):
            q = curve[sample_idx]

            fk_flat = (
                np.array(self.robot.forward_kinematics(q)).astype(float).reshape(-1)
            )
            joint_positions = fk_flat.reshape(self.n_links + 1, 3)

            for link_idx in range(self.n_links):
                out[sample_idx, link_idx] = 0.5 * (
                    joint_positions[link_idx] + joint_positions[link_idx + 1]
                )

        return out


@dataclass
class CasadiFKGaussians:
    """
    Compute arbitrary Gaussian sampling points along robot links.

    Parameters
    ----------
    robot : object
        Robot object exposing `forward_kinematics(q)` and `get_n_joints()`.
    n_links : int
        Number of links in the active chain.
    """

    robot: object
    n_links: int
    fk_fun: cas.Function = field(init=False)

    def __post_init__(self):
        """
        Prebuild a CasADi forward-kinematics function for repeated evaluation.
        """
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
        """
        Compute Gaussian sampling points along links for all trajectory samples.

        Parameters
        ----------
        curve : np.ndarray
            Joint trajectory of shape `(num_samples, n_joints)`.
        gaussian_specs : list[tuple[int, float]]
            Gaussian definitions as `(link_idx, t)`, where `t in [0, 1]`
            interpolates along one link segment.

        Returns
        -------
        np.ndarray
            Gaussian points with shape `(num_samples, n_total_gaussians, 3)`.

        Raises
        ------
        ValueError
            If any link index or interpolation factor is invalid.
        """
        curve = np.asarray(curve, dtype=float)
        num_samples = curve.shape[0]
        n_total_gaussians = len(gaussian_specs)

        if n_total_gaussians == 0:
            return np.zeros((num_samples, 0, 3), dtype=float)

        points = np.zeros((num_samples, n_total_gaussians, 3), dtype=float)

        for sample_idx in range(num_samples):
            fk_flat = np.array(self.fk_fun(curve[sample_idx])).astype(float).reshape(-1)
            joint_positions = fk_flat.reshape(self.n_links + 1, 3)

            for gaussian_idx, (link_idx, t) in enumerate(gaussian_specs):
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
                points[sample_idx, gaussian_idx] = (1.0 - t) * p0 + t * p1

        return points


class GaussianModel:
    """
    Interface for covariance lookup associated with robot Gaussian samples.
    """

    def cov(self, link_idx: int) -> np.ndarray:
        """
        Return the covariance matrix associated with one link.

        Parameters
        ----------
        link_idx : int
            Link index.

        Returns
        -------
        np.ndarray
            Covariance matrix of shape `(3, 3)`.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SharedCovariance(GaussianModel):
    """
    Covariance model shared by all links.
    """

    cov_mat: np.ndarray

    def cov(self, link_idx: int) -> np.ndarray:
        """
        Return the shared covariance matrix.
        """
        return self.cov_mat


@dataclass(frozen=True)
class PerLinkCovariances(GaussianModel):
    """
    Link-dependent covariance model.
    """

    covs: np.ndarray

    def cov(self, link_idx: int) -> np.ndarray:
        """
        Return the covariance matrix associated with one link.
        """
        return self.covs[link_idx]


@dataclass(frozen=True)
class EllipsoidFactory:
    """
    Convert covariance matrices into ellipsoid radii and orientations.

    Parameters
    ----------
    n_std : float, default=2.0
        Number of standard deviations used to define the ellipsoid radius.
    """

    n_std: float = 2.0

    def cov_to_ellipsoid(self, cov: np.ndarray):
        """
        Convert a covariance matrix into ellipsoid geometry.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix of shape `(3, 3)`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(radii, quat_wxyz)` where:
            - `radii` has shape `(3,)`
            - `quat_wxyz` is the ellipsoid orientation quaternion in WXYZ order
        """
        eigvals, eigvecs = np.linalg.eigh(cov)

        radii = self.n_std * np.sqrt(np.maximum(eigvals, 1e-12))

        rotation_matrix = eigvecs
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] *= -1

        quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return radii, quat_wxyz
