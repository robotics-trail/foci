from abc import ABC, abstractmethod
from typing import Any

class BaseRobot(ABC):
    """
    Minimal common interface for all robot models.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of optimization variables."""
        raise NotImplementedError

    @abstractmethod
    def f_task(self, q: Any) -> Any:
        """
        Map configuration q to the task-space point used by the goal cost.
        """
        raise NotImplementedError

    @abstractmethod
    def collision_points(self, q: Any) -> Any:
        """
        Return collision evaluation points with shape (n_points, 3).
        """
        raise NotImplementedError


    @abstractmethod
    def forward_kinematics(self, q: Any) -> Any:
        """
        Return forward kinematics of the robot.
        """
        raise NotImplementedError
    
    @abstractmethod
    def collision_covariances(self): 
        raise NotImplementedError

    @abstractmethod
    def collision_covariances_online(self, q: Any) -> Any: 
        raise NotImplementedError

    def joint_limits(self) -> list[tuple[float, float]] | None:
        """
        Optional joint limits.

        Robots without limits can return None.
        """
        return None
    
    def validate_q(self, q: Any) -> None:
        if q.shape[0] != self.n_dof:
            raise ValueError(
                f"Expected q of length {self.n_dof}, got {q.shape[0]}"
            )
    