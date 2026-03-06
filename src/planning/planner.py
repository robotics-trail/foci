import numpy as np
import casadi as cas

from typing import List, Tuple

from ompl import base as ob
from ompl import geometric as og

from src.core.robot_loader import ManipulatorRobotURDF
from src.optim.solver import create_solver, create_multiple_gaussians_solver
from src.splines.bspline import BSpline
from src.planning.initializer import RRTStarInitializer


class Planner:
    def __init__(
        self,
        urdf_file: str,
        root_link: str,
        tip_link: str,
        obstacle_positions: np.ndarray,
        obstacle_covs: np.ndarray,
        robot_cov: np.ndarray,  # (3, 3) / (n_links, 3, 3)
        num_control_points: int = 8,
        num_samples: int = 30,
        weights: dict = {"jerk": 0.1, "goal": 40.0, "obstacle": 40.0},
        wmax: float = 1.0,
        vmax: float = 1.0,
        amax: float = 1.0,
        ignore_link_indices: list = None,
    ):

        # --- Robot ---
        self.robot = ManipulatorRobotURDF(urdf_file, root_link, tip_link)
        self.n_joints = self.robot.get_n_joints()
        self.n_links = self.robot.get_n_links()

        self.joint_limits = self.robot.get_joint_limits()
        self.chain_links = self.robot.get_links()

        # --- Planning params ---
        self.num_control_points = num_control_points
        self.num_samples = num_samples
        self.weights = weights
        self.wmax = wmax
        self.vmax = vmax
        self.amax = amax
        self.obstacle_positions = obstacle_positions
        self.obstacle_covs = obstacle_covs
        self.robot_cov = robot_cov

        ignore_link_indices = ignore_link_indices or []
        self.ignore_link_indices = sorted(set(ignore_link_indices))
        self.active_link_indices = [
            i for i in range(self.n_links) if i not in ignore_link_indices
        ]
        self.n_active_links = len(self.active_link_indices)

        # --- Precompute covs ---
        if robot_cov.ndim == 2:
            self.multiple_gaussians = False
            covs_sum = self.obstacle_covs + robot_cov  # (n_obstacles, 3, 3)

        elif robot_cov.ndim == 3 and robot_cov.shape[0] == self.n_links:
            self.multiple_gaussians = True
            covs_sum = np.zeros(
                (self.n_active_links, obstacle_covs.shape[0], 3, 3)
            )  # (n_active_links, n_obstacles, 3, 3)

            for k, link_idx in enumerate(self.active_link_indices):
                covs_sum[k, :, :, :] = self.obstacle_covs + robot_cov[link_idx]

        else:
            raise ValueError(
                f"robot_cov must be (3,3) or (n_active_links,3,3) with n_links={self.n_active_links}. Got {robot_cov.shape}"
            )

        self.covs_det = np.linalg.det(covs_sum)
        self.covs_inv = np.linalg.inv(covs_sum)

        self.solver, self.lbg, self.ubg, self.convolution_functor = (
            self._create_solver()
        )

    def _create_solver(self):
        solver, lbg, ubg, convolution_functor = create_solver(
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

        return solver, lbg, ubg, convolution_functor

    def plan(self, theta_start: np.ndarray, ee_goal: np.ndarray):

        rrt_star = RRTStarInitializer(self.robot)
        init_guess = rrt_star.generate_initial_path(
            theta_start, ee_goal, self.num_control_points
        )

        params_val = np.concatenate((theta_start, ee_goal))

        res = self.solver(x0=init_guess, lbg=self.lbg, ubg=self.ubg, p=params_val)

        print(res)

        control_points_opt = (
            np.array(res["x"]).reshape(self.n_joints, self.num_control_points).T
        )
        bspline = BSpline(control_points_opt)
        opt_curve = bspline.spline_eval(self.num_samples)

        return opt_curve

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


class MultipleGaussiansPlanner:
    def __init__(
        self,
        urdf_file: str,
        root_link: str,
        tip_link: str,
        obstacle_positions: np.ndarray,
        obstacle_covs: np.ndarray,
        robot_cov: np.ndarray,  # (3, 3) / (n_links, 3, 3)
        gaussians_per_link: List[Tuple],
        num_control_points: int = 8,
        num_samples: int = 30,
        weights: dict = {"jerk": 0.1, "goal": 40.0, "obstacle": 40.0},
        wmax: float = 1.0,
        vmax: float = 1.0,
        amax: float = 1.0,
        ignore_link_indices: list = None,
    ):

        # --- Robot ---
        self.robot = ManipulatorRobotURDF(urdf_file, root_link, tip_link)
        self.n_joints = self.robot.get_n_joints()
        self.n_links = self.robot.get_n_links()

        self.joint_limits = self.robot.get_joint_limits()
        self.chain_links = self.robot.get_links()

        # --- Planning params ---
        self.num_control_points = num_control_points
        self.num_samples = num_samples
        self.weights = weights
        self.wmax = wmax
        self.vmax = vmax
        self.amax = amax
        self.obstacle_positions = obstacle_positions
        self.obstacle_covs = obstacle_covs
        self.robot_cov = robot_cov
        self.gaussians_per_link = gaussians_per_link
        self.gaussian_specs = self._build_gaussian_specs(self.gaussians_per_link)
        self.n_total_gaussians = len(self.gaussian_specs)

        ignore_link_indices = ignore_link_indices or []
        self.ignore_link_indices = sorted(set(ignore_link_indices))
        self.active_link_indices = [
            i for i in range(self.n_links) if i not in ignore_link_indices
        ]
        self.n_active_links = len(self.active_link_indices)

        # --- Precompute covs ---
        if robot_cov.ndim == 2:
            self.multiple_gaussians = False
            covs_sum = self.obstacle_covs + robot_cov  # (n_obstacles, 3, 3)

        elif robot_cov.ndim == 3 and robot_cov.shape[0] == self.n_links:
            self.multiple_gaussians = True
            covs_sum = np.zeros(
                (self.n_active_links, obstacle_covs.shape[0], 3, 3)
            )  # (n_active_links, n_obstacles, 3, 3)

            for k, link_idx in enumerate(self.active_link_indices):
                covs_sum[k, :, :, :] = self.obstacle_covs + robot_cov[link_idx]

        else:
            raise ValueError(
                f"robot_cov must be (3,3) or (n_active_links,3,3) with n_links={self.n_active_links}. Got {robot_cov.shape}"
            )

        self.covs_det = np.linalg.det(covs_sum)
        self.covs_inv = np.linalg.inv(covs_sum)

        self.solver, self.lbg, self.ubg, self.convolution_functor = (
            self._create_solver()
        )

    def _build_gaussian_specs(self, gaussians_per_link):
        # TODO: REALIZAR COMPROBACIONES
        gaussian_specs = []
        for link_idx, t_values in gaussians_per_link:
            for t in t_values:
                gaussian_specs.append((link_idx, t))

        return gaussian_specs

    def _create_solver(self):
        solver, lbg, ubg, convolution_functor = create_multiple_gaussians_solver(
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

        return solver, lbg, ubg, convolution_functor

    def plan(self, theta_start: np.ndarray, ee_goal: np.ndarray):

        rrt_star = RRTStarInitializer(self.robot)
        init_guess = rrt_star.generate_initial_path(
            theta_start, ee_goal, self.num_control_points
        )

        params_val = np.concatenate((theta_start, ee_goal))

        res = self.solver(x0=init_guess, lbg=self.lbg, ubg=self.ubg, p=params_val)

        print(res)

        control_points_opt = (
            np.array(res["x"]).reshape(self.n_joints, self.num_control_points).T
        )
        bspline = BSpline(control_points_opt)
        opt_curve = bspline.spline_eval(self.num_samples)

        return opt_curve

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
