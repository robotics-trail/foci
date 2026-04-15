from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import open3d as o3d
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from src.core.robot_loader import ManipulatorRobotURDF
from src.planning.utils import EEGoal


@dataclass
class RRTStarConfig:
    """
    Configuration parameters for the OMPL RRT* initializer.
    """

    solve_time: float = 3.0
    voxel_size: float = 0.10
    random_seed: Optional[int] = 42


class PathInitializer(ABC):
    """
    Abstract interface for initial trajectory generators.
    """

    @abstractmethod
    def generate_initial_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ) -> np.ndarray:
        """
        Generate an initial path flattened in row-major order.
        """
        raise NotImplementedError


class RRTStarInitializer(PathInitializer):
    """
    Initial path generator based on OMPL RRT* in joint space.

    The goal is defined in Cartesian end-effector space through `EEGoal`.
    """

    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        obstacle_means: np.ndarray,
        gaussians_per_link,
        config: Optional[RRTStarConfig] = None,
    ):
        self.robot = robot
        self.n_joints = robot.get_n_joints()
        self.joint_limits = robot.get_joint_limits()

        self.obstacle_means = obstacle_means
        self.gaussians_per_link = gaussians_per_link

        self.config = RRTStarConfig() if config is None else config

        self.occupancy_map = self._build_occupancy_map(
            self.obstacle_means, self.config.voxel_size
        )

        if self.config.random_seed is not None:
            ou.RNG.setSeed(self.config.random_seed)

    def generate_initial_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
        threshold: float = 0.01,
    ) -> np.ndarray:
        """
        Generate an initial path for the optimizer.

        Parameters
        ----------
        start : np.ndarray
            Initial joint configuration with shape (n_joints,).
        goal : np.ndarray
            Cartesian end-effector target with shape (3,).
        num_control_points : int
            Number of path samples requested for the optimizer initialization.

        Returns
        -------
        np.ndarray
            Flattened path with shape (num_control_points * n_joints,).
        """
        space, space_info = self._setup_space()
        start_state = self._create_state(space, start)
        goal_region = EEGoal(space_info, self.robot, goal, threshold)

        return self._plan_rrt_star(
            space_info,
            start_state,
            goal_region,
            num_control_points,
        )

    def _build_occupancy_map(self, obstacle_means, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obstacle_means)

        return o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=voxel_size
        )

    def _plan_rrt_star(
        self,
        space_info: ob.SpaceInformation,
        start_state: ob.State,
        goal_region: EEGoal,
        num_control_points: int,
    ) -> np.ndarray:
        """
        Run OMPL RRT* and interpolate the resulting path.

        If no solution is found, a constant initialization repeating the start
        configuration is returned.
        """
        problem = ob.ProblemDefinition(space_info)
        problem.addStartState(start_state)
        problem.setGoal(goal_region)

        planner = og.RRTstar(space_info)
        planner.setProblemDefinition(problem)
        planner.setup()

        solved = planner.solve(self.config.solve_time)

        if solved:
            path = problem.getSolutionPath()
            path.interpolate(num_control_points)
            return self._path_to_numpy(path)

        return np.tile(
            start_state_to_numpy(start_state, self.n_joints), (num_control_points, 1)
        ).flatten(order="C")

    def _setup_space(self):
        """
        Create OMPL joint-space bounds and state validity checker.
        """
        space = ob.RealVectorStateSpace(self.n_joints)
        bounds = ob.RealVectorBounds(self.n_joints)

        for joint_idx, (lower, upper) in enumerate(self.joint_limits):
            bounds.setLow(joint_idx, float(lower))
            bounds.setHigh(joint_idx, float(upper))

        space.setBounds(bounds)

        space_info = ob.SpaceInformation(space)
        space_info.setStateValidityChecker(
            ob.StateValidityCheckerFn(self._is_state_valid)
        )
        space_info.setup()

        return space, space_info

    def _create_state(self, space, config: np.ndarray):
        """
        Convert a NumPy joint vector into an OMPL state.
        """
        state = ob.State(space)
        for joint_idx in range(self.n_joints):
            state[joint_idx] = float(config[joint_idx])
        return state

    def _is_state_valid(self, state) -> bool:
        """
        Check whether a state is inside joint limits.
        """
        q = np.array([state[joint_idx] for joint_idx in range(self.n_joints)])

        if not self._within_joint_limits(q):
            return False

        if self._robot_in_collision(q):
            return False

        return True

    def _within_joint_limits(self, q: np.ndarray) -> bool:
        for joint_idx, (lower, upper) in enumerate(self.joint_limits):
            if q[joint_idx] < lower or q[joint_idx] > upper:
                return False

        return True

    def _robot_in_collision(self, q: np.ndarray):
        fk = self.robot.forward_kinematics(q)
        fk = np.array(fk.full()).reshape(-1)
        points = fk.reshape(-1, 3)

        sampled_points = []

        for link_idx, gaussian_t_list in self.gaussians_per_link:
            p0 = points[link_idx]
            p1 = points[link_idx + 1]

            for t in gaussian_t_list:
                point = (1.0 - t) * p0 + t * p1
                sampled_points.append(point)

        if len(sampled_points) == 0:
            return False

        sampled_points = np.asarray(sampled_points, dtype=float)

        included = self.occupancy_map.check_if_included(
            o3d.utility.Vector3dVector(sampled_points)
        )

        return np.any(np.asarray(included, dtype=bool))

    def _path_to_numpy(self, path: og.PathGeometric) -> np.ndarray:
        """
        Flatten an OMPL path into shape (num_points * n_joints,).
        """
        states = []
        for path_idx in range(path.getStateCount()):
            state = path.getState(path_idx)
            q = np.array(
                [float(state[joint_idx]) for joint_idx in range(self.n_joints)]
            )
            states.append(q)

        return np.array(states).flatten(order="C")


def start_state_to_numpy(state, n_joints: int) -> np.ndarray:
    """
    Convert one OMPL state into a NumPy joint vector.
    """
    return np.array([float(state[joint_idx]) for joint_idx in range(n_joints)])
