from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PlannerWeights:
    """
    Relative weights of the trajectory optimization objective.

    Attributes
    ----------
    jerk : float
        Weight of the smoothness / jerk regularization term.
    goal : float
        Weight of the end-effector goal tracking term.
    obstacle : float
        Weight of the obstacle avoidance term.
    """

    jerk: float = 0.1
    goal: float = 40.0
    obstacle: float = 40.0

    def as_dict(self) -> dict:
        """Return weights in the dictionary format expected by the solver layer."""
        return {
            "jerk": self.jerk,
            "goal": self.goal,
            "obstacle": self.obstacle,
        }


@dataclass
class PlannerLimits:
    """
    Motion limits used by the trajectory optimizer.

    Attributes
    ----------
    wmax : float
        Joint-velocity bound used in hull constraints.
    vmax : float
        Cartesian speed estimate used to scale spline time.
    amax : float
        Joint-acceleration bound used in hull constraints.
    """

    wmax: float = 1.0
    vmax: float = 1.0
    amax: float = 1.0


@dataclass
class ProblemConfig:
    """
    Full configuration of one planning problem.

    Parameters
    ----------
    urdf_file : str
        Path to the robot URDF.
    root_link : str
        Root link of the active kinematic chain.
    tip_link : str
        Tip link of the active kinematic chain.
    obstacle_positions : np.ndarray
        Obstacle centers with shape (n_obstacles, 3).
    obstacle_covs : np.ndarray
        Obstacle covariance matrices with shape (n_obstacles, 3, 3).
    robot_cov : np.ndarray
        Robot covariance model, either shared with shape (3, 3) or per-link
        with shape (n_links, 3, 3).
    theta_start : np.ndarray
        Initial joint configuration with shape (n_joints,).
    ee_goal : np.ndarray
        Desired end-effector goal position with shape (3,).
    initializer_solve_time: float
        Desired solve time for initializer.
    random_seed: int
        Desired random seed for initializer.
    """

    urdf_file: str
    root_link: str
    tip_link: str

    obstacle_positions: np.ndarray
    obstacle_covs: np.ndarray
    robot_cov: np.ndarray

    theta_start: np.ndarray
    ee_goal: np.ndarray

    num_control_points: int = 8
    num_samples: int = 30

    weights: PlannerWeights = field(default_factory=PlannerWeights)
    limits: PlannerLimits = field(default_factory=PlannerLimits)

    ignore_link_indices: List[int] = field(default_factory=list)
    gaussians_per_link: Optional[List[Tuple[int, List[float]]]] = None

    initializer_solve_time: float = field(default=3.0)
    random_seed: int = field(default=42)

    @property
    def use_multiple_gaussians(self) -> bool:
        """Whether the problem uses explicit Gaussian samples along links."""
        return self.gaussians_per_link is not None and len(self.gaussians_per_link) > 0
