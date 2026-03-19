from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class PlannerWeights:
    jerk: float = 0.1
    goal: float = 40.0
    obstacle: float = 40.0

    def as_dict(self) -> dict:
        return {
            "jerk": self.jerk,
            "goal": self.goal,
            "obstacle": self.obstacle,
        }


@dataclass
class PlannerLimits:
    wmax: float = 1.0
    vmax: float = 1.0
    amax: float = 1.0


@dataclass
class ProblemConfig:
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

    @property
    def use_multiple_gaussians(self) -> bool:
        return self.gaussians_per_link is not None and len(self.gaussians_per_link) > 0
