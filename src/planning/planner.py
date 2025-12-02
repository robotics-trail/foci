import numpy as np
import casadi as cas

from src.core.robot_loader import ManipulatorRobotURDF
from src.optim.solver import create_solver
from src.splines.bspline import BSpline


class Planner:
    def __init__(
        self,
        urdf_file: str,
        root_link: str,
        tip_link: str,
        obstacle_positions: np.ndarray,
        obstacle_covs: np.ndarray,
        robot_cov: np.ndarray,
        num_control_points: int = 8,
        num_samples: int = 30,
        weights: dict = {"jerk": 0.1, "goal": 40.0, "obstacle": 40.0},
        wmax: float = 1.0,
        vmax: float = 1.0,
        amax: float = 1.0,
    ):

        # --- Robot ---
        self.robot = ManipulatorRobotURDF(urdf_file, root_link, tip_link)
        self.n_joints = self.robot.get_n_joints()
        self.n_links = self.robot.get_n_links()

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

        # --- Precompute covs ---
        covs_sum = obstacle_covs + robot_cov
        self.covs_det = np.array([np.linalg.det(c) for c in covs_sum])
        self.covs_inv = np.array([np.linalg.inv(c) for c in covs_sum])

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
            num_samples=self.num_samples,
            weights=self.weights,
            wmax=self.wmax,
            vmax=self.vmax,
            amax=self.amax,
        )

        return solver, lbg, ubg, convolution_functor

    def plan(self, theta_start: np.ndarray, ee_goal: np.ndarray):
        # TODO: CHANGE FOR A* INIT GUESS
        init_guess = np.tile(theta_start, (self.num_control_points, 1)).flatten(
            order="C"
        )

        params_val = np.concatenate((theta_start, ee_goal))
        res = self.solver(x0=init_guess, lbg=self.lbg, ubg=self.ubg, p=params_val)

        # print(res)

        control_points_opt = np.array(res["x"]).reshape(
            self.num_control_points, self.n_joints
        )

        bspline = BSpline(control_points_opt)
        opt_curve = bspline.spline_eval(self.num_samples)

        return opt_curve
