"""
Trajectory planner interfaces built on top of the optimization layer.

This module provides:
- a shared base planner with common robot/problem setup
- a standard midpoint-based planner
- a planner using multiple Gaussian samples along selected links
"""

from typing import List, Tuple

import time
import numpy as np

from src.core.robot_loader import ManipulatorRobotURDF
from src.optim.solver import create_multiple_gaussians_solver, create_solver
from src.planning.config import ProblemConfig
from src.planning.initializer import RRTStarInitializer, RRTStarConfig
from src.splines.bspline import BSpline


class RRTStarPlanner:
    """
    Baseline planner using only OMPL RRT*.
    """

    def __init__(self, config: ProblemConfig, initializer_config: RRTStarConfig):
        self.config = config
        self.initializer_config = initializer_config
        self.robot = ManipulatorRobotURDF(
            config.urdf_file,
            config.root_link,
            config.tip_link,
        )
        self.n_joints = self.robot.get_n_joints()
        self.num_control_points = config.num_control_points
        self.num_samples = config.num_samples

    def plan(
        self,
        theta_start: np.ndarray = None,
        ee_goal: np.ndarray = None,
        return_timings: bool = True,
    ):
        theta_start = self.config.theta_start if theta_start is None else theta_start
        ee_goal = self.config.ee_goal if ee_goal is None else ee_goal

        initializer = RRTStarInitializer(self.robot, config=self.initializer_config)

        t0 = time.perf_counter()
        control_points_flat = initializer.generate_initial_path(
            theta_start, ee_goal, self.num_control_points, threshold=0.01
        )
        t1 = time.perf_counter()

        control_points = control_points_flat.reshape(
            self.num_control_points, self.n_joints
        )

        # Para comparar con tu método final, lo convertimos también en trayectoria muestreada
        bspline = BSpline(control_points)
        trajectory = bspline.spline_eval(self.num_samples)

        timings = {
            "rrt_time": t1 - t0,
            "total_time": t1 - t0,
        }

        if return_timings:
            return trajectory, timings

        return trajectory


class BasePlanner:
    """
    Base class for optimization-based trajectory planners.

    This class owns:
    - the robot model
    - common planning parameters
    - covariance preprocessing
    - the optimization solve loop

    Subclasses only need to implement `_create_solver()`.
    """

    def __init__(self, config: ProblemConfig, initializer_config: RRTStarConfig):
        """
        Parameters
        ----------
        config : ProblemConfig
            Full planning problem definition.
        initializer_config : RRTStarConfig
            Initializer definition.
        """
        self.config = config
        self.initializer_config = initializer_config

        # --- Robot ---
        self.robot = ManipulatorRobotURDF(
            config.urdf_file,
            config.root_link,
            config.tip_link,
        )
        self.n_joints = self.robot.get_n_joints()
        self.n_links = self.robot.get_n_links()

        self.joint_limits = self.robot.get_joint_limits()
        self.chain_links = self.robot.get_links()

        # --- Planning parameters ---
        self.num_control_points = config.num_control_points
        self.num_samples = config.num_samples
        self.weights = config.weights.as_dict()

        self.wmax = config.limits.wmax
        self.vmax = config.limits.vmax
        self.amax = config.limits.amax

        self.obstacle_positions = config.obstacle_positions
        self.obstacle_covs = config.obstacle_covs
        self.robot_cov = config.robot_cov

        self.ignore_link_indices = sorted(set(config.ignore_link_indices))
        self.active_link_indices = [
            link_idx
            for link_idx in range(self.n_links)
            if link_idx not in self.ignore_link_indices
        ]
        self.n_active_links = len(self.active_link_indices)

        # --- Precomputed obstacle/robot covariance terms ---
        (
            self.multiple_gaussians,
            self.covs_det,
            self.covs_inv,
        ) = self._precompute_covariances()

        # --- Solver ---
        self.solver, self.lbg, self.ubg, self.convolution_functor = (
            self._create_solver()
        )

    def _precompute_covariances(self):
        """
        Precompute determinants and inverses of obstacle+robot covariance sums.

        Returns
        -------
        tuple
            `(multiple_gaussians, covs_det, covs_inv)` where:
            - `multiple_gaussians` indicates whether the robot covariance is
              link-dependent
            - `covs_det` contains covariance determinants
            - `covs_inv` contains covariance inverses

        Raises
        ------
        ValueError
            If `robot_cov` does not have shape `(3, 3)` or `(n_links, 3, 3)`.
        """
        robot_cov = self.robot_cov

        if robot_cov.ndim == 2:
            combined_covs = self.obstacle_covs + robot_cov
            return False, np.linalg.det(combined_covs), np.linalg.inv(combined_covs)

        if robot_cov.ndim == 3 and robot_cov.shape[0] == self.n_links:
            combined_covs = np.zeros(
                (self.n_active_links, self.obstacle_covs.shape[0], 3, 3)
            )

            for local_idx, link_idx in enumerate(self.active_link_indices):
                combined_covs[local_idx] = self.obstacle_covs + robot_cov[link_idx]

            return True, np.linalg.det(combined_covs), np.linalg.inv(combined_covs)

        raise ValueError(
            "robot_cov must have shape (3, 3) or (n_links, 3, 3). "
            f"Got {robot_cov.shape}."
        )

    def _build_initializer(self) -> RRTStarInitializer:
        """
        Create the initializer used to generate the optimization warm start.

        Returns
        -------
        RRTStarInitializer
            Joint-space RRT* initializer.
        """
        return RRTStarInitializer(self.robot, self.initializer_config)

    def plan(
        self,
        theta_start: np.ndarray = None,
        ee_goal: np.ndarray = None,
        return_timings: bool = True,
    ) -> np.ndarray:
        """
        Solve the trajectory optimization problem.

        Parameters
        ----------
        theta_start : np.ndarray, optional
            Initial joint configuration of shape `(n_joints,)`.
            If omitted, `config.theta_start` is used.
        ee_goal : np.ndarray, optional
            Target end-effector position of shape `(3,)`.
            If omitted, `config.ee_goal` is used.

        Returns
        -------
        np.ndarray
            Optimized spline samples with shape `(num_samples, n_joints)`.
        """
        theta_start = self.config.theta_start if theta_start is None else theta_start
        ee_goal = self.config.ee_goal if ee_goal is None else ee_goal

        initializer = self._build_initializer()

        t0 = time.perf_counter()
        initial_guess = initializer.generate_initial_path(
            theta_start,
            ee_goal,
            self.num_control_points,
        )
        t1 = time.perf_counter()

        params_val = np.concatenate((theta_start, ee_goal))

        t2 = time.perf_counter()
        result = self.solver(
            x0=initial_guess,
            lbg=self.lbg,
            ubg=self.ubg,
            p=params_val,
        )
        t3 = time.perf_counter()

        optimal_control_points = (
            np.array(result["x"]).reshape(self.n_joints, self.num_control_points).T
        )

        bspline = BSpline(optimal_control_points)
        trajectory = bspline.spline_eval(self.num_samples)

        timings = {
            "initializer_rrt_time": t1 - t0,
            "solver_time": t3 - t2,
            "total_time": (t1 - t0) + (t3 - t2),
        }

        if return_timings:
            return trajectory, timings

        return trajectory

    def _create_solver(self):
        """
        Create the CasADi solver for the current planner type.

        Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class Planner(BasePlanner):
    """
    Standard planner using one collision evaluation point per active link.

    The obstacle term is built from link midpoints sampled along the trajectory.
    """

    def _create_solver(self):
        """
        Create the midpoint-based optimization solver.

        Returns
        -------
        tuple
            `(solver, lbg, ubg, convolution_functor)`.
        """
        return create_solver(
            robot=self.robot,
            num_control_points=self.num_control_points,
            obstacle_means=self.obstacle_positions,
            covs_det=self.covs_det,
            covs_inv=self.covs_inv,
            multiple_gaussians=self.multiple_gaussians,
            active_link_indices=self.active_link_indices,
            num_samples=self.num_samples,
            weights=self.weights,
            wmax=self.wmax,
            vmax=self.vmax,
            amax=self.amax,
        )


class MultipleGaussiansPlanner(BasePlanner):
    """
    Planner using multiple Gaussian evaluation points along selected links.

    Gaussian samples are defined by `(link_idx, t)` pairs, where `t in [0, 1]`
    indicates the interpolation factor along a link segment.
    """

    def __init__(self, config: ProblemConfig, initializer_config: RRTStarConfig):
        """
        Parameters
        ----------
        config : ProblemConfig
            Planning problem configuration. Must include `gaussians_per_link`.

        Raises
        ------
        ValueError
            If the configuration does not define Gaussian samples.
        """
        if not config.use_multiple_gaussians:
            raise ValueError(
                "MultipleGaussiansPlanner requires config.gaussians_per_link."
            )

        self.gaussians_per_link = config.gaussians_per_link
        self.gaussian_specs = self._build_gaussian_specs(self.gaussians_per_link)
        self.n_total_gaussians = len(self.gaussian_specs)

        super().__init__(config, initializer_config)

    def _build_gaussian_specs(
        self,
        gaussians_per_link: List[Tuple[int, List[float]]],
    ) -> List[Tuple[int, float]]:
        """
        Flatten grouped Gaussian definitions into a list of point specifications.

        Parameters
        ----------
        gaussians_per_link : list[tuple[int, list[float]]]
            Per-link Gaussian definitions. Each entry contains a link index and
            a list of interpolation parameters along that link.

        Returns
        -------
        list[tuple[int, float]]
            Flat list of `(link_idx, t)` pairs.
        """
        gaussian_specs = []
        for link_idx, t_values in gaussians_per_link:
            for t in t_values:
                gaussian_specs.append((link_idx, t))

        return gaussian_specs

    def _create_solver(self):
        """
        Create the multi-Gaussian optimization solver.

        Returns
        -------
        tuple
            `(solver, lbg, ubg, convolution_functor)`.
        """
        return create_multiple_gaussians_solver(
            robot=self.robot,
            num_control_points=self.num_control_points,
            obstacle_means=self.obstacle_positions,
            covs_det=self.covs_det,
            covs_inv=self.covs_inv,
            multiple_gaussians=self.multiple_gaussians,
            active_link_indices=self.active_link_indices,
            gaussian_specs=self.gaussian_specs,
            num_samples=self.num_samples,
            weights=self.weights,
            wmax=self.wmax,
            vmax=self.vmax,
            amax=self.amax,
        )
