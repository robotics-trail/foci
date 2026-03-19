import numpy as np

from typing import List, Tuple

from src.core.robot_loader import ManipulatorRobotURDF
from src.optim.solver import create_solver, create_multiple_gaussians_solver
from src.splines.bspline import BSpline
from src.planning.initializer import RRTStarInitializer
from src.planning.config import ProblemConfig


class BasePlanner:
    def __init__(self, config: ProblemConfig):
        self.config = config

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

        # --- Planning params ---
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
            i for i in range(self.n_links) if i not in self.ignore_link_indices
        ]
        self.n_active_links = len(self.active_link_indices)

        self.multiple_gaussians, self.covs_det, self.covs_inv = self._precompute_covs()

        self.solver, self.lbg, self.ubg, self.convolution_functor = (
            self._create_solver()
        )

    def _precompute_covs(self):
        robot_cov = self.robot_cov

        if robot_cov.ndim == 2:
            multiple_gaussians = False
            covs_sum = self.obstacle_covs + robot_cov

        elif robot_cov.ndim == 3 and robot_cov.shape[0] == self.n_links:
            multiple_gaussians = True
            covs_sum = np.zeros(
                (self.n_active_links, self.obstacle_covs.shape[0], 3, 3)
            )

            for k, link_idx in enumerate(self.active_link_indices):
                covs_sum[k] = self.obstacle_covs + robot_cov[link_idx]

        else:
            raise ValueError(
                f"robot_cov must be (3,3) or (n_links,3,3). Got {robot_cov.shape}"
            )

        return multiple_gaussians, np.linalg.det(covs_sum), np.linalg.inv(covs_sum)

    def plan(self, theta_start: np.ndarray = None, ee_goal: np.ndarray = None):
        theta_start = self.config.theta_start if theta_start is None else theta_start
        ee_goal = self.config.ee_goal if ee_goal is None else ee_goal

        rrt_star = RRTStarInitializer(self.robot)
        init_guess = rrt_star.generate_initial_path(
            theta_start,
            ee_goal,
            self.num_control_points,
        )

        params_val = np.concatenate((theta_start, ee_goal))
        res = self.solver(x0=init_guess, lbg=self.lbg, ubg=self.ubg, p=params_val)

        control_points_opt = (
            np.array(res["x"]).reshape(self.n_joints, self.num_control_points).T
        )

        bspline = BSpline(control_points_opt)
        return bspline.spline_eval(self.num_samples)

    def _path_to_numpy(self, path):
        states = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            q = np.array([state[j] for j in range(self.n_joints)])
            states.append(q)

        return np.array(states).flatten(order="C")

    def _is_state_valid(self, state):
        q = np.array([state[i] for i in range(self.n_joints)])

        for i, (low, high) in enumerate(self.joint_limits):
            if q[i] < low or q[i] > high:
                return False

        return True

    def _create_sovler(self):
        raise NotImplementedError


class Planner(BasePlanner):
    def _create_solver(self):
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
    def __init__(self, config: ProblemConfig):
        if not config.use_multiple_gaussians:
            raise ValueError(
                "MultipleGaussiansPlanner requires config.gaussians_per_link."
            )

        self.gaussians_per_link = config.gaussians_per_link
        self.gaussian_specs = self._build_gaussian_specs(self.gaussians_per_link)
        self.n_total_gaussians = len(self.gaussian_specs)

        super().__init__(config)

    def _build_gaussian_specs(self, gaussians_per_link):
        gaussian_specs = []
        for link_idx, t_values in gaussians_per_link:
            for t in t_values:
                gaussian_specs.append((link_idx, t))
        return gaussian_specs

    def _create_solver(self):
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
