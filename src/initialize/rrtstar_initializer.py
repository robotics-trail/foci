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


# How many path samples per control point are used to fit the initial guess.
# The fit must be over-determined, otherwise the spline interpolates the path
# and can overshoot around corners.
_PATH_SAMPLES_PER_CONTROL_POINT: int = 5


def fit_control_points(
    path_points: np.ndarray,
    num_control_points: int,
    start: np.ndarray,
    regularization: float = 1e-9,
) -> np.ndarray:
    """
    Fit B-spline control points to a configuration-space path.

    A uniform cubic B-spline does not interpolate its control points: with
    control points p, the curve starts at (p_0 + 4 p_1 + p_2) / 6.  Using the
    path samples directly as control points therefore produces a guess whose
    first curve point is not `start`, i.e. one that violates the start equality
    constraint of the NLP.  Here the control points are instead obtained from a
    least-squares fit of the curve to the path, with curve(0) == start imposed
    as a hard equality constraint.

    Path samples are assumed to be uniformly spaced in the spline parameter.

    Parameters
    ----------
    path_points : np.ndarray
        Path samples of shape (num_path_samples, n_dof).
    num_control_points : int
        Number of control points of the resulting spline.
    start : np.ndarray
        Start configuration of shape (n_dof,), imposed exactly on curve(0).
    regularization : float, default=1e-9
        Tikhonov term keeping the normal equations non-singular.

    Returns
    -------
    np.ndarray
        Control points of shape (num_control_points, n_dof).
    """
    path_points = np.asarray(path_points, dtype=float)
    start = np.asarray(start, dtype=float).reshape(1, -1)

    n_dof = path_points.shape[1]

    if start.shape[1] != n_dof:
        raise ValueError(
            f"start has {start.shape[1]} entries but the path has {n_dof} columns."
        )

    # Only the shape of the control points matters to build the basis.
    bspline = BSpline(np.zeros((num_control_points, n_dof)))

    basis = bspline.basis_matrix(
        bspline.sample_parameters(path_points.shape[0])
    )                                                    # (num_path_samples, N)
    start_basis = bspline.basis_matrix([0.0])            # (1, N)

    normal_matrix = basis.T @ basis + regularization * np.eye(num_control_points)

    # KKT system of  min ||basis @ P - path||^2  s.t.  start_basis @ P == start
    kkt_matrix = np.block(
        [
            [normal_matrix, start_basis.T],
            [start_basis, np.zeros((1, 1))],
        ]
    )
    kkt_rhs = np.vstack((basis.T @ path_points, start))

    solution = np.linalg.solve(kkt_matrix, kkt_rhs)

    return solution[:num_control_points, :]


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
            environment.obstacle_means_at(0),
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

        path_points, success = self._solve_until_solution(
            space_info=space_info,
            start_state=start_state,
            goal_region=goal_region,
            num_control_points=num_control_points,
        )

        t1 = perf_counter()

        if path_points is None:
            # Constant guess: every control point equal, so curve(0) == start.
            control_points = np.tile(start, (num_control_points, 1))
            success = False
        else:
            control_points = fit_control_points(
                path_points,
                num_control_points,
                start,
            )

        bspline = BSpline(control_points)
        trajectory = np.array(bspline.spline_eval(num_control_points))


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
        """Run RRT* and return the solution path as an array of samples."""
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
                path.interpolate(
                    _PATH_SAMPLES_PER_CONTROL_POINT * num_control_points
                )
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

        if hasattr(points, "full"):
            points = points.full()

        points = np.asarray(points, dtype=float).reshape(-1, 3)

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