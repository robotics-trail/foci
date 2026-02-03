import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod

from typing import Optional

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from src.core.robot_loader import ManipulatorRobotURDF
from src.planning.utils import EEGoal


# @dataclass
# class RRTConfig:

#     max_planning_time: float = 2.0
#     range: float = 0.5
#     goal_bias: float = 0.05
#     random_seed: Optional[int] = 42
#     rewire_factor: float = 1.1
#     informed_sampling: bool = True
#     optimization_objective: str = "path_length"


class PathInitializer(ABC):
    @abstractmethod
    def generate_initial_path(
        self, start: np.ndarray, goal: np.ndarray, num_points: int
    ) -> np.ndarray:
        pass


class RRTStarInitializer(PathInitializer):
    def __init__(self, robot: ManipulatorRobotURDF):
        self.robot = robot
        self.n_joints = robot.get_n_joints()
        self.joint_limits = robot.get_joint_limits()

        ou.RNG.setSeed(42)

    def generate_initial_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ):

        space, si = self._setup_space()
        start_state = self._create_state(space, start)
        goal_region = EEGoal(si, self.robot, goal)

        path = self._plan_rrt_star(si, start_state, goal_region, num_control_points)

        return path

    def _plan_rrt_star(
        self,
        si: ob.SpaceInformation,
        start: ob.State,
        goal: EEGoal,
        num_control_points: int,
    ):

        pdef = ob.ProblemDefinition(si)
        pdef.addStartState(start)
        pdef.setGoal(goal)

        planner = og.RRTstar(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(3.0)

        if solved:
            path = pdef.getSolutionPath()
            path.interpolate(num_control_points)
            init_guess = self._path_to_numpy(path)

        else:
            init_guess = np.tile(start, (num_control_points, 1)).flatten(order="C")

        return init_guess

    def _setup_space(self):
        space = ob.RealVectorStateSpace(self.n_joints)
        bounds = ob.RealVectorBounds(self.n_joints)

        for i, (low, high) in enumerate(self.joint_limits):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))

        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_state_valid))
        si.setup()

        return space, si

    def _create_state(self, space, config):
        state = ob.State(space)

        for i in range(self.n_joints):
            state[i] = config[i]

        return state

    def _is_state_valid(self, state):
        q = np.array([state[i] for i in range(self.n_joints)])

        for i, (low, high) in enumerate(self.joint_limits):
            if q[i] < low or q[i] > high:
                return False

        return True

    def _path_to_numpy(self, path: og.PathGeometric) -> np.ndarray:
        states = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = np.array([float(state[j]) for j in range(self.n_joints)])
            states.append(q)
        return np.array(states).flatten(order="C")
