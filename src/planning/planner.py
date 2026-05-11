from time import perf_counter
import numpy as np

from src.optimization.problem import build_problem
from src.optimization.solver import create_solver
from src.planning.result import PlanningResult
from src.planning.joints import JointGroups
from src.initialize.straight_line_initializer import StraightLineInitializer
from src.splines.bspline import BSpline

class Planner:
    """
    Unified trajectory optimization planner.

    This planner works with any robot implementing the BaseRobot API.

    Supported robots:
    - ManipulatorRobot
    - DroneRobot
    - future mobile robots

    Supported methods:
    - FOCI
    - future CHOMP/STOMP integration
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
    ):
        self.robot = robot
        self.environment = environment

        if initializer is None: 
            self.initializer = StraightLineInitializer()

        else: 
            self.initializer = initializer

        self.num_control_points = num_control_points
        self.num_samples = num_samples

        self.joint_groups = joint_groups

        self.weights = weights
        self.vmax = vmax


    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        """
        Run trajectory optimization.

        Parameters
        ----------
        start :
            Initial robot configuration.

        goal :
            Goal task-space position.

        Returns
        -------
        PlanningResult
        """
        total_start = perf_counter()

        # ==========================================================
        # Initializer
        # ==========================================================

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

        # ==========================================================
        # Build optimization problem
        # ==========================================================

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

        solver = create_solver(nlp)

        build_time = perf_counter() - build_start

        # ==========================================================
        # Initial guess
        # ==========================================================

        if initial_control_points is None:
            x0 = np.tile(start, (self.num_control_points, 1))
        else:
            x0 = initial_control_points

        x0 = np.asarray(x0).reshape(-1, order="F")

        # ==========================================================
        # Solve NLP
        # ==========================================================

        solve_start = perf_counter()

        params = np.concatenate([
            np.asarray(start).reshape(-1),
            np.asarray(goal).reshape(-1),
        ])

        solution = solver(
            x0=x0,
            p=params,
            lbg=lbg,
            ubg=ubg,
        )

        solve_time = perf_counter() - solve_start

        # ==========================================================
        # Extract solution
        # ==========================================================

        control_points = np.array(
            solution["x"]
        ).reshape(self.num_control_points, self.robot.n_dof, order="F")

        spline = BSpline(control_points)

        trajectory = np.array(
            spline.spline_eval(self.num_samples)
        )

        total_time = perf_counter() - total_start

        # ==========================================================
        # Solver stats
        # ==========================================================

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