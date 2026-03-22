import numpy as np
from ompl import base as ob

from src.core.robot_loader import ManipulatorRobotURDF


class EEGoal(ob.GoalRegion):
    """
    OMPL goal region defined in Cartesian end-effector space.

    The planner operates in joint space, but the terminal goal is expressed as
    a 3D end-effector target. This class evaluates the distance to that target
    using the robot forward kinematics.
    """

    def __init__(
        self,
        space_information,
        robot: ManipulatorRobotURDF,
        goal: np.ndarray,
        threshold: float = 0.05,
    ):
        super().__init__(space_information)
        self.robot = robot
        self.goal = goal
        self.setThreshold(threshold)

    def distanceGoal(self, state) -> float:
        """
        Compute Euclidean distance from the state's end-effector position to the goal.
        """
        q = np.array([state[i] for i in range(self.robot.get_n_joints())])
        ee_position = np.array(self.robot.get_ee_endpoint(q)).astype(float).reshape(-1)
        return float(np.linalg.norm(ee_position - self.goal))
