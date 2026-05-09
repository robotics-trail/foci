from __future__ import annotations


from time import perf_counter
from typing import Optional

import numpy as np
import open3d as o3d

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from src.splines.bspline import BSpline
from src.initialize.initializer import InitializerResult, PathInitializer

    

class RRTStarInitializer(PathInitializer):
    """
    RRT* initializer in configuration space.

    The goal is defined in task space using:

        robot.f_task(q)

    Collision is checked using:

        robot.collision_points(q)

    This makes the initializer independent of whether the robot is a manipulator,
    drone, or future mobile robot.
    """

    def __init__(self, voxel_size: float = 0.10, goal_threshold: float = 0.01, random_seed: Optional[int] = 42, check_interval: float = 0.2, max_time: float | None = None):
        
        self.voxel_size: float = voxel_size
        self.goal_threshold: float = goal_threshold
        self.check_interval: float = check_interval
        self.max_time: float = max_time

        if random_seed is not None:
            ou.RNG.setSeed(random_seed)

    def initialize(
        self,
        robot,
        environment,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ) -> InitializerResult:
        t0 = perf_counter()

        self.robot = robot
        self.environment = environment
        self.n_dof = robot.n_dof
        self.joint_limits = self._joint_limits(robot)
        self.occupancy_map = self._build_occupancy_map(
            environment.obstacle_means,
            self.voxel_size,
        )

        space, space_info = self._setup_space()
        start_state = self._create_state(space, start)
        goal_region = TaskSpaceGoal(
            space_information=space_info,
            robot=robot,
            goal=np.asarray(goal, dtype=float),
            threshold=self.goal_threshold,
        )

        control_points, success = self._solve_until_solution(
            space_info=space_info,
            start_state=start_state,
            goal_region=goal_region,
            num_control_points=num_control_points,
        )

        t1 = perf_counter()

        if control_points is None:
            control_points = np.tile(start, (num_control_points, 1))
            success = False

        trajectory = BSpline(control_points).spline_eval(num_control_points)

        return InitializerResult(
            control_points=control_points,
            trajectory=trajectory,
            success=success,
            timings={"initializer": t1 - t0},
            metadata={
                "type": "rrtstar",
                "max_time": self.max_time,
                "check_interval": self.check_interval,
            },
        )
    
    def _solve_until_solution(
        self,
        space_info: ob.SpaceInformation,
        start_state: ob.State,
        goal_region: ob.GoalRegion,
        num_control_points: int,
    ) -> tuple[np.ndarray | None, bool]:
        problem = ob.ProblemDefinition(space_info)
        problem.addStartState(start_state)
        problem.setGoal(goal_region)

        planner = og.RRTstar(space_info)
        planner.setProblemDefinition(problem)
        planner.setup()

        t0 = perf_counter()

        while True:
            solved = planner.solve(self.check_interval)

            if solved:
                path = problem.getSolutionPath()
                path.interpolate(num_control_points)
                return self._path_to_numpy(path), True

            if self.max_time is not None:
                if perf_counter() - t0 >= self.max_time:
                    return None, False

    def _setup_space(self):
        space = ob.RealVectorStateSpace(self.n_dof)
        bounds = ob.RealVectorBounds(self.n_dof)

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

    def _joint_limits(self, robot) -> list[tuple[float, float]]:
        limits = robot.joint_limits()

        if limits is None:
            return [(-np.inf, np.inf) for _ in range(robot.n_dof)]

        if len(limits) != robot.n_dof:
            raise ValueError(
                f"joint_limits must contain {robot.n_dof} entries, "
                f"got {len(limits)}."
            )

        clean_limits = []

        for lower, upper in limits:
            lower = -1e6 if np.isneginf(lower) else lower
            upper = 1e6 if np.isposinf(upper) else upper
            clean_limits.append((lower, upper))

        return clean_limits

    def _create_state(self, space, q: np.ndarray):
        state = ob.State(space)

        for joint_idx in range(self.n_dof):
            state[joint_idx] = float(q[joint_idx])

        return state

    def _is_state_valid(self, state) -> bool:
        q = np.array(
            [state[joint_idx] for joint_idx in range(self.n_dof)],
            dtype=float,
        )

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

    def _robot_in_collision(self, q: np.ndarray) -> bool:
        points = self.robot.collision_points(q)

        included = self.occupancy_map.check_if_included(
            o3d.utility.Vector3dVector(points)
        )

        return bool(np.any(np.asarray(included, dtype=bool)))

    def _build_occupancy_map(self, obstacle_means, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obstacle_means)

        return o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,
            voxel_size=voxel_size,
        )

    def _path_to_numpy(self, path: og.PathGeometric) -> np.ndarray:
        states = []

        for path_idx in range(path.getStateCount()):
            state = path.getState(path_idx)

            q = np.array(
                [float(state[joint_idx]) for joint_idx in range(self.n_dof)],
                dtype=float,
            )

            states.append(q)

        return np.asarray(states, dtype=float)

class TaskSpaceGoal(ob.GoalRegion):
    """
    OMPL goal region defined through robot.f_task(q).
    """

    def __init__(
        self,
        space_information,
        robot,
        goal: np.ndarray,
        threshold: float = 0.01,
    ):
        super().__init__(space_information)

        self.robot = robot
        self.goal = np.asarray(goal, dtype=float).reshape(3)

        self.setThreshold(float(threshold))

    def distanceGoal(self, state) -> float:
        q = np.array(
            [state[i] for i in range(self.robot.n_dof)],
            dtype=float,
        )

        point = self.robot.f_task(q)

        if hasattr(point, "full"):
            point = point.full()

        point = np.asarray(point, dtype=float).reshape(3)

        return float(np.linalg.norm(point - self.goal))