import numpy as np

from ompl import base as ob

from src.core.robot_loader import ManipulatorRobotURDF


class EEGoal(ob.GoalRegion):
    def __init__(
        self, si, robot: ManipulatorRobotURDF, goal: np.ndarray, threshold: float = 0.05
    ):
        super().__init__(si)
        self.robot = robot
        self.goal = goal
        self.setThreshold(threshold)

    def distanceGoal(self, state: np.ndarray):
        q = np.array([state[i] for i in range(self.robot.get_n_joints())])
        ee = np.array(self.robot.get_ee_endpoint(q)).astype(float)
        return np.linalg.norm(ee - self.goal)
