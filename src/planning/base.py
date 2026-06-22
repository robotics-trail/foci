"""Abstract base class for trajectory optimisation planners.

Both the static (Planner) and online (OnlinePlanner) variants share the same
__init__ and plan() logic.  The only concrete difference is which
build_problem function is called, so subclasses implement _build_problem().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np

from src.initialize.uniform_initializer import UniformControlPointsInitializer
from src.optimization.solver import create_solver
from src.planning.joints import JointGroups
from src.planning.result import PlanningResult
from src.splines.bspline import BSpline


class BasePlanner(ABC):
    """Abstract trajectory optimisation planner.

    Works with any robot implementing the BaseRobot API and any environment
    implementing BaseGaussianEnvironment.

    Parameters
    ----------
    robot:
        Robot model (e.g. ManipulatorRobot, DroneRobot).
    environment:
        Obstacle environment.
    joint_groups:
        Grouping of joints used when building the optimisation problem.
    initializer:
        Strategy for producing an initial guess. Defaults to
        UniformControlPointsInitializer when None.
    num_control_points:
        Number of B-spline control points.
    num_samples:
        Number of trajectory samples evaluated during optimisation.
    weights:
        Cost-term weights forwarded to build_problem.
    vmax:
        Maximum joint velocity.
    linear_solver:
        IPOPT linear solver backend (e.g. "mumps", "ma27").
    """

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
    ) -> None:
        self.robot = robot
        self.environment = environment
        self.joint_groups = joint_groups
        self.initializer = (
            initializer if initializer is not None else UniformControlPointsInitializer()
        )
        self.num_control_points = num_control_points
        self.num_samples = num_samples
        self.weights = weights
        self.vmax = vmax
        self.linear_solver = linear_solver

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_problem(self) -> tuple:
        """Call the appropriate build_problem function and return
        (nlp, lbg, ubg, callbacks).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_parameter_vector(
        self, start: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Concatenate start and goal into the NLP parameter vector."""
        return np.concatenate([
            np.asarray(start, dtype=float).reshape(-1),
            np.asarray(goal, dtype=float).reshape(-1),
        ])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan(self, start: np.ndarray, goal: np.ndarray) -> PlanningResult:
        """Run trajectory optimisation from start to goal.

        Parameters
        ----------
        start:
            Initial robot configuration.
        goal:
            Goal task-space position.

        Returns
        -------
        PlanningResult
        """
        total_start = perf_counter()

        # --- Initialisation ------------------------------------------
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

        # --- Build optimisation problem -------------------------------
        build_start = perf_counter()

        nlp, lbg, ubg, callbacks = self._build_problem()
        self._casadi_callbacks = callbacks

        solver = create_solver(
            nlp, options={"ipopt.linear_solver": self.linear_solver}
        )

        build_time = perf_counter() - build_start

        # --- Initial guess -------------------------------------------
        x0 = (
            np.tile(start, (self.num_control_points, 1))
            if initial_control_points is None
            else initial_control_points
        )
        x0 = np.asarray(x0, dtype=float).reshape(-1, order="F")

        # --- Solve NLP -----------------------------------------------
        solve_start = perf_counter()

        solution = solver(
            x0=x0,
            p=self._build_parameter_vector(start, goal),
            lbg=lbg,
            ubg=ubg,
        )

        solve_time = perf_counter() - solve_start

        # --- Extract solution ----------------------------------------
        control_points = np.array(solution["x"]).reshape(
            self.num_control_points, self.robot.n_dof, order="F"
        )

        trajectory = np.array(BSpline(control_points).spline_eval(self.num_samples))

        total_time = perf_counter() - total_start

        # --- Return result -------------------------------------------
        stats = solver.stats()

        return PlanningResult(
            trajectory=trajectory,
            control_points=control_points,
            initial_trajectory=initial_trajectory,
            success=stats.get("success", False),
            timings={
                "initializer": init_time,
                "build": build_time,
                "solve": solve_time,
                "total": total_time,
            },
            solver_stats=stats,
        )