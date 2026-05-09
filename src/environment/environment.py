from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianEnvironment:
    """
    Gaussian obstacle environment.

    Obstacles are represented as a Gaussian mixture:

        obstacle_means: shape (n_obstacles, 3)
        obstacle_covariances: shape (n_obstacles, 3, 3)
    """

    obstacle_means: np.ndarray
    obstacle_covariances: np.ndarray

    def __post_init__(self):
        self.obstacle_means = np.asarray(self.obstacle_means, dtype=float)
        self.obstacle_covariances = np.asarray(
            self.obstacle_covariances,
            dtype=float,
        )

        if self.obstacle_means.ndim != 2 or self.obstacle_means.shape[1] != 3:
            raise ValueError(
                "obstacle_means must have shape (n_obstacles, 3)."
            )

        if (
            self.obstacle_covariances.ndim != 3
            or self.obstacle_covariances.shape[1:] != (3, 3)
        ):
            raise ValueError(
                "obstacle_covariances must have shape (n_obstacles, 3, 3)."
            )

        if self.obstacle_covariances.shape[0] != self.obstacle_means.shape[0]:
            raise ValueError(
                "obstacle_means and obstacle_covariances must contain "
                "the same number of obstacles."
            )

    @property
    def n_obstacles(self) -> int:
        return self.obstacle_means.shape[0]

    def convolved_covariances(
        self,
        robot_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return determinants and inverses of obstacle + robot covariances.
        """
        robot_covariance = np.asarray(robot_covariance, dtype=float)

        if robot_covariance.shape != (3, 3):
            raise ValueError("robot_covariance must have shape (3, 3).")

        covariances = self.obstacle_covariances + robot_covariance

        return (
            np.linalg.det(covariances),
            np.linalg.inv(covariances),
        )