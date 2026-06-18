from time import perf_counter

import numpy as np

from src.initialize.uniform_initializer import UniformControlPointsInitializer
from src.optimization.problem_online import build_problem
from src.optimization.solver import create_solver
from src.planning.joints import JointGroups
from src.planning.result import PlanningResult
from src.splines.bspline import BSpline


class OnlinePlanner:
    def __init__(
        self,
        robot,
        environment,
        joint_groups: JointGroups,
        initializer=None,
        num_control_points: int = 8,
        num_samples: int = 30,
        weights: dict[str, float] | None = None,
        vmax: float = 1.0,
        linear_solver: str = "mumps",
    ):
        self.robot = robot
        self.environment = environment
        self.linear_solver = linear_solver

        if initializer is None:
            self.initializer = UniformControlPointsInitializer()
        else:
            self.initializer = initializer

        self.num_control_points = num_control_points
        self.num_samples = num_samples
        self.joint_groups = joint_groups
        self.weights = weights
        self.vmax = vmax

    @staticmethod
    def _obstacle_covariances_to_row_major(covariances: np.ndarray) -> np.ndarray:
        covariances = np.asarray(covariances, dtype=float)

        if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
            raise ValueError(
                "environment.obstacle_covariances must have shape "
                f"(num_obstacles, 3, 3), got {covariances.shape}."
            )

        return covariances.reshape(covariances.shape[0], 9, order="C")

    def _build_parameter_vector(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(start, dtype=float).reshape(-1),
                np.asarray(goal, dtype=float).reshape(-1),

            ]
        )

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        total_start = perf_counter()

        init_start = perf_counter()

        init_result = self.initializer.initialize(
            robot=self.robot,
            environment=self.environment,
            start=start,
            goal=goal,
            num_control_points=self.num_control_points,
        )

        initial_control_points = init_result.control_points
        initial_trajectory = init_result.trajectory
        init_time = perf_counter() - init_start

        build_start = perf_counter()

        nlp, lbg, ubg, callbacks = build_problem(
            robot=self.robot,
            environment=self.environment,
            num_control_points=self.num_control_points,
            num_samples=self.num_samples,
            joint_groups=self.joint_groups,
            weights=self.weights,
            vmax=self.vmax,
        )

        self._casadi_callbacks = callbacks

        solver = create_solver(
            nlp,
            options={"ipopt.linear_solver": self.linear_solver},
        )

        build_time = perf_counter() - build_start

        if initial_control_points is None:
            x0 = np.tile(start, (self.num_control_points, 1))
        else:
            x0 = initial_control_points

        x0 = np.asarray(x0, dtype=float).reshape(-1, order="F")
        params = self._build_parameter_vector(start=start, goal=goal)

        solve_start = perf_counter()

        solution = solver(
            x0=x0,
            p=params,
            lbg=lbg,
            ubg=ubg,
        )

        solve_time = perf_counter() - solve_start

        control_points = np.array(solution["x"]).reshape(
            self.num_control_points,
            self.robot.n_dof,
            order="F",
        )

        spline = BSpline(control_points)
        trajectory = np.array(spline.spline_eval(self.num_samples))

        total_time = perf_counter() - total_start

        stats = solver.stats()
        success = stats.get("success", False)

        return PlanningResult(
            trajectory=trajectory,
            control_points=control_points,
            initial_trajectory=initial_trajectory,
            success=success,
            timings={
                "initializer": init_time,
                "build": build_time,
                "solve": solve_time,
                "total": total_time,
            },
            solver_stats=stats,
        )
