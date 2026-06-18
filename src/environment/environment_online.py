from dataclasses import dataclass

import numpy as np

from src.environment.obstacle import BaseObstacle


@dataclass
class GaussianEnvironmentOnline:
    """
    Gaussian obstacle environment supporting static and mobile obstacles.

    obstacles: list[BaseObstacle]
        Each obstacle implements obstacle_mean(k) -> (3,)
        and obstacle_covariance(k) -> (3, 3).
    """

    obstacles: list[BaseObstacle]

    @property
    def n_obstacles(self) -> int:
        return len(self.obstacles)

    def obstacle_means_at(self, k: int) -> np.ndarray:
        """Return obstacle means at sample k. Shape: (n_obstacles, 3)."""
        return np.stack([o.obstacle_mean(k) for o in self.obstacles], axis=0)

    def obstacle_covariances_at(self, k: int) -> np.ndarray:
        """Return obstacle covariances at sample k. Shape: (n_obstacles, 3, 3)."""
        return np.stack([o.obstacle_covariance(k) for o in self.obstacles], axis=0)

    def build_per_sample_means(self, num_samples: int) -> np.ndarray:
        """
        Return per-sample means for all obstacles.

        Shape: (num_samples, n_obstacles * 3), row-major per obstacle.
        Each row k contains the flattened means of all obstacles at sample k:
            [mean0_x, mean0_y, mean0_z, mean1_x, ...]
        """
        rows = []
        for k in range(num_samples):
            means_k = self.obstacle_means_at(k)       # (n_obstacles, 3)
            rows.append(means_k.reshape(-1))           # (n_obstacles * 3,)
        return np.stack(rows, axis=0)                  # (num_samples, n_obstacles * 3)

    def build_per_sample_covariances(self, num_samples: int) -> np.ndarray:
        """
        Return per-sample covariances for all obstacles.

        Shape: (num_samples, n_obstacles * 9), row-major per obstacle covariance.
        Each row k contains the flattened covariances of all obstacles at sample k:
            [cov0_00, cov0_01, ..., cov0_22, cov1_00, ...]
        """
        rows = []
        for k in range(num_samples):
            covs_k = self.obstacle_covariances_at(k)   # (n_obstacles, 3, 3)
            rows.append(covs_k.reshape(-1))             # (n_obstacles * 9,)
        return np.stack(rows, axis=0)                   # (num_samples, n_obstacles * 9)