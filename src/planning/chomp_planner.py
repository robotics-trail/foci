"""
CHOMP (Covariant Hamiltonian Optimization for Motion Planning) planner.

Implements the same interface as BasePlanner / MultipleGaussiansPlanner so that
benchmarking requires changing only one line in the test script.

CHOMP minimizes:
    U(xi) = lambda * f_smooth(xi) + f_obs(xi)

where:
    f_smooth(xi) = (1/2) * xi^T @ A^T @ A @ xi      (finite-difference jerk)
    f_obs(xi)    = sum over waypoints and gaussians of the Gaussian obstacle cost

The obstacle cost uses the same Gaussian mixture representation already present
in ProblemConfig (obstacle_means, obstacle_covs, robot_cov), so results are
directly comparable with MultipleGaussiansPlanner.

References
----------
Ratliff et al., "CHOMP: Gradient Optimization Techniques for Efficient
Motion Planning", ICRA 2009.
"""

from typing import List, Optional, Tuple

import time
import numpy as np
import casadi as cas

from src.core.robot_loader import ManipulatorRobotURDF
from src.planning.config import ProblemConfig
from src.planning.initializer import RRTStarConfig, RRTStarInitializer
from src.splines.bspline import BSpline


# ---------------------------------------------------------------------------
# Finite-difference smoothness matrix
# ---------------------------------------------------------------------------

def _build_finite_difference_matrix(n: int, order: int = 2) -> np.ndarray:
    """
    Build finite-difference matrix A of shape (n - order, n).

    A^T @ A gives the smoothness (acceleration) penalty matrix K.

    Parameters
    ----------
    n : int
        Number of waypoints.
    order : int
        Finite-difference order. 1 = velocity, 2 = acceleration, 3 = jerk.
    """
    D = np.eye(n)
    for _ in range(order):
        diff = np.diff(np.eye(D.shape[0]), axis=0)
        D = diff @ D
    return D  # shape (n - order, n)


# ---------------------------------------------------------------------------
# Gaussian obstacle cost and gradient
# ---------------------------------------------------------------------------

def _gaussian_cost(
    point: np.ndarray,          # (3,)
    means: np.ndarray,          # (n_obs, 3)
    covs_inv: np.ndarray,       # (n_obs, 3, 3)
    covs_det: np.ndarray,       # (n_obs,)
) -> Tuple[float, np.ndarray]:
    """
    Compute the sum of Gaussian obstacle costs at a 3-D workspace point,
    plus the gradient with respect to that point.

    Cost = sum_k  1/sqrt((2pi)^3 * det(Sigma_k))
                  * exp(-0.5 * (p - mu_k)^T @ Sigma_k^{-1} @ (p - mu_k))

    Returns
    -------
    cost : float
    grad : np.ndarray, shape (3,)
    """
    cost = 0.0
    grad = np.zeros(3)

    n_obs = means.shape[0]
    for k in range(n_obs):
        diff = point - means[k]                          # (3,)
        normalizer = 1.0 / np.sqrt((2 * np.pi) ** 3 * covs_det[k])
        exponent = -0.5 * diff @ covs_inv[k] @ diff
        g = normalizer * np.exp(exponent)

        cost += g
        grad += g * (-covs_inv[k] @ diff)               # chain rule

    return cost, grad


# ---------------------------------------------------------------------------
# Forward kinematics helpers — use robot's cached CasADi FK functions
# ---------------------------------------------------------------------------

def _eval_link_origin(robot, link_name: str, theta: np.ndarray) -> np.ndarray:
    """
    Return the 3-D world-frame origin of `link_name` at joint config `theta`.

    Uses the precomputed CasADi FK function stored in robot.link_fk_funcs.
    Only the first `link_joint_counts[link_name]` joints are passed in,
    matching how forward_kinematics() works internally.
    """
    fk_fn = robot.link_fk_funcs[link_name]
    n_q   = robot.link_joint_counts[link_name]
    q_dm  = cas.DM(theta[:n_q])
    T     = np.array(fk_fn(q_dm))          # (4, 4) as numpy
    return T[:3, 3]                         # (3,)


def _get_link_positions(
    robot,
    theta: np.ndarray,
    gaussian_specs: List[Tuple[int, float]],
) -> np.ndarray:
    """
    Evaluate workspace positions for each (link_idx, t) Gaussian spec.

    Interpolates between consecutive link origins (same semantics as
    MultipleGaussiansPlanner's Gaussian placement along a link segment).

    Parameters
    ----------
    robot : ManipulatorRobotURDF
    theta : np.ndarray, shape (n_joints,)
    gaussian_specs : list of (link_idx, t)

    Returns
    -------
    positions : np.ndarray, shape (n_gaussians, 3)
    """
    links = robot.get_links()   # ordered list of link names

    # Cache origins so each link is evaluated once per waypoint
    origin_cache: dict = {}

    def get_origin(idx: int) -> np.ndarray:
        if idx not in origin_cache:
            origin_cache[idx] = _eval_link_origin(robot, links[idx], theta)
        return origin_cache[idx]

    positions = []
    for link_idx, t in gaussian_specs:
        if link_idx == 0:
            p = get_origin(0)
        else:
            p_prev = get_origin(link_idx - 1)
            p_curr = get_origin(link_idx)
            p = (1.0 - t) * p_prev + t * p_curr
        positions.append(p)

    return np.array(positions)  # (n_gaussians, 3)


def _numerical_jacobian(
    robot,
    link_name: str,
    theta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute a numerical (3 x n_joints) position Jacobian for `link_name`
    via central finite differences on the CasADi FK.

    Only columns up to link_joint_counts[link_name] are non-zero;
    the rest are left as zero (joints after this link don't affect it).
    """
    n_q   = robot.n_joints
    p0    = _eval_link_origin(robot, link_name, theta)   # (3,)
    J     = np.zeros((3, n_q))

    n_q_link = robot.link_joint_counts[link_name]
    for j in range(n_q_link):
        th_p = theta.copy(); th_p[j] += eps
        th_m = theta.copy(); th_m[j] -= eps
        p_p  = _eval_link_origin(robot, link_name, th_p)
        p_m  = _eval_link_origin(robot, link_name, th_m)
        J[:, j] = (p_p - p_m) / (2.0 * eps)

    return J  # (3, n_joints)


# ---------------------------------------------------------------------------
# CHOMPPlanner
# ---------------------------------------------------------------------------

class CHOMPPlanner:
    """
    CHOMP trajectory optimizer with the same planning interface as
    MultipleGaussiansPlanner.

    Parameters
    ----------
    config : ProblemConfig
        Full planning problem definition (same object used for your planner).
    initializer_config : RRTStarConfig
        RRT* warm-start configuration (same object used for your planner).
    n_iter : int
        Maximum number of gradient-descent iterations.
    learning_rate : float
        Step size for the gradient update.
    lambda_smooth : float
        Weight on the smoothness term. Overrides config.weights.jerk if set.
    fd_order : int
        Finite-difference order for the smoothness matrix (2 = acceleration).
    tol : float
        Convergence tolerance on the cost change.
    """

    def __init__(
        self,
        config: ProblemConfig,
        initializer_config: RRTStarConfig,
        n_iter: int = 200,
        learning_rate: float = 0.01,
        lambda_smooth: Optional[float] = None,
        fd_order: int = 2,
        tol: float = 1e-6,
    ):
        self.config = config
        self.initializer_config = initializer_config
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.tol = tol

        # ---- Robot --------------------------------------------------------
        self.robot = ManipulatorRobotURDF(
            config.urdf_file,
            config.root_link,
            config.tip_link,
        )
        self.n_joints = self.robot.get_n_joints()
        self.n_links = self.robot.get_n_links()

        # ---- Planning parameters ------------------------------------------
        self.num_control_points = config.num_control_points
        self.num_samples = config.num_samples
        self.weights = config.weights.as_dict()

        self.obstacle_positions = config.obstacle_positions   # (n_obs, 3)
        self.obstacle_covs = config.obstacle_covs             # (n_obs, 3, 3)
        self.robot_cov = config.robot_cov

        self.ignore_link_indices = sorted(set(config.ignore_link_indices))
        self.active_link_indices = [
            i for i in range(self.n_links) if i not in self.ignore_link_indices
        ]

        # Gaussian specs for collision evaluation (same as MultipleGaussiansPlanner)
        self.gaussians_per_link = config.gaussians_per_link
        self.gaussian_specs: List[Tuple[int, float]] = []
        for link_idx, t_values in self.gaussians_per_link:
            for t in t_values:
                self.gaussian_specs.append((link_idx, t))

        # ---- Smoothness weight --------------------------------------------
        self.lambda_smooth = (
            lambda_smooth if lambda_smooth is not None
            else self.weights.get("jerk", 1e-5)
        )

        # ---- Covariances --------------------------------------------------
        # Use a single robot_cov (mean over links if per-link) for simplicity
        if self.robot_cov.ndim == 3:
            mean_robot_cov = self.robot_cov[self.active_link_indices].mean(axis=0)
        else:
            mean_robot_cov = self.robot_cov

        combined_covs = self.obstacle_covs + mean_robot_cov   # (n_obs, 3, 3)
        self.covs_det = np.linalg.det(combined_covs)          # (n_obs,)
        self.covs_inv = np.linalg.inv(combined_covs)          # (n_obs, 3, 3)

        # ---- Finite-difference smoothness matrix --------------------------
        # Operates on the full waypoint trajectory (num_samples x n_joints)
        N = self.num_samples
        A = _build_finite_difference_matrix(N, order=fd_order)  # (N-order, N)
        self.K = A.T @ A                                         # (N, N) smoothness kernel

        # ---- Joint limits for projection ---------------------------------
        raw_limits = self.robot.get_joint_limits()   # list of (lower, upper)
        self.joint_lb = np.array([l for l, _ in raw_limits], dtype=float)
        self.joint_ub = np.array([u for _, u in raw_limits], dtype=float)
        # Replace inf with large finite values so clipping is always valid
        self.joint_lb = np.where(np.isinf(self.joint_lb), -2 * np.pi, self.joint_lb)
        self.joint_ub = np.where(np.isinf(self.joint_ub),  2 * np.pi, self.joint_ub)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smoothness_cost_and_grad(
        self, xi: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        f_smooth = 0.5 * sum_j  xi_j^T @ K @ xi_j   (sum over joints)

        grad shape: (num_samples, n_joints)
        """
        Kxi  = self.K @ xi                    # (N, n_joints)
        cost = 0.5 * float(np.sum(xi * Kxi))  # scalar
        grad = Kxi                             # (N, n_joints)
        return cost, grad

    def _obstacle_cost_and_grad(
        self, xi: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        For each waypoint evaluate the Gaussian cost at each Gaussian spec
        position, then back-project the workspace gradient to joint space
        via the Jacobian.

        xi : (num_samples, n_joints)
        Returns cost (float) and grad (num_samples, n_joints).
        """
        N, n_joints = xi.shape
        total_cost = 0.0
        total_grad = np.zeros_like(xi)

        for i in range(N):
            theta_i = xi[i]

            # Workspace positions for this configuration
            try:
                ws_positions = _get_link_positions(
                    self.robot, theta_i, self.gaussian_specs
                )
            except Exception:
                # FK failed — skip waypoint
                continue

            for g_idx, (link_idx, _) in enumerate(self.gaussian_specs):
                p = ws_positions[g_idx]
                c, grad_ws = _gaussian_cost(
                    p,
                    self.obstacle_positions,
                    self.covs_inv,
                    self.covs_det,
                )
                total_cost += c

                # Back-project to joint space via numerical Jacobian (3 x n_joints)
                link_name = self.robot.get_links()[link_idx]
                J = _numerical_jacobian(self.robot, link_name, theta_i)  # (3, n_joints)
                total_grad[i] += J.T @ grad_ws

        return total_cost, total_grad


    def _goal_cost_and_grad(
        self,
        xi: np.ndarray,
        ee_goal: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Penalise the distance between the end-effector at the last waypoint
        and ee_goal, and back-project to joint space via the numerical Jacobian.

        cost    = 0.5 * ||FK(xi[-1]) - ee_goal||^2
        grad[-1] = J_ee^T @ (FK(xi[-1]) - ee_goal)   (all other rows = 0)
        """
        grad = np.zeros_like(xi)

        theta_last = xi[-1]
        q_dm       = cas.DM(theta_last)
        ee_pos     = np.array(self.robot.get_ee_endpoint(q_dm)).flatten()  # (3,)
        error      = ee_pos - ee_goal                                        # (3,)
        cost       = 0.5 * float(error @ error)

        tip_link  = self.robot.tip_link
        J_ee      = _numerical_jacobian(self.robot, tip_link, theta_last)   # (3, n_joints)
        grad[-1]  = J_ee.T @ error

        return cost, grad

    def _fix_endpoints(
        self,
        xi: np.ndarray,
        theta_start: np.ndarray,
        theta_end: np.ndarray,
    ) -> np.ndarray:
        """Clamp start and end waypoints to their fixed values."""
        xi[0] = theta_start
        xi[-1] = theta_end
        return xi

    # ------------------------------------------------------------------
    # Public API  (mirrors BasePlanner.plan)
    # ------------------------------------------------------------------

    def plan(
        self,
        theta_start: np.ndarray = None,
        ee_goal: np.ndarray = None,
        return_timings: bool = True,
    ):
        """
        Run CHOMP to optimize a collision-free trajectory.

        Parameters
        ----------
        theta_start : np.ndarray, optional
            Start joint configuration. Falls back to config.theta_start.
        ee_goal : np.ndarray, optional
            Goal end-effector position. Falls back to config.ee_goal.
        return_timings : bool
            If True, return (trajectory, timings) matching BasePlanner.plan.

        Returns
        -------
        trajectory : np.ndarray, shape (num_samples, n_joints)
        timings : dict  (only when return_timings=True)
            Keys: 'initializer_rrt_time', 'solver_time', 'total_time'
        """
        theta_start = self.config.theta_start if theta_start is None else theta_start
        ee_goal = self.config.ee_goal if ee_goal is None else ee_goal

        # ---- Warm start via RRT* (same as your planners) -----------------
        initializer = RRTStarInitializer(
            self.robot,
            self.obstacle_positions,
            self.gaussians_per_link,
            self.initializer_config,
        )

        t0 = time.perf_counter()
        control_points_flat = initializer.generate_initial_path(
            theta_start, ee_goal, self.num_control_points
        )
        t1 = time.perf_counter()

        control_points = control_points_flat.reshape(
            self.num_control_points, self.n_joints
        )

        # Sample into a dense waypoint trajectory to pass to CHOMP
        bspline_init = BSpline(control_points)
        xi = bspline_init.spline_eval(self.num_samples)   # (N, n_joints)

        # The start is fixed; the end floats so the goal term can pull it.
        theta_start_fixed = theta_start.copy()

        # Goal weight from PlannerWeights (same key your solver uses)
        lambda_goal = self.weights.get("goal", 100.0)

        # ---- CHOMP gradient descent ---------------------------------------
        t2 = time.perf_counter()

        prev_cost = np.inf
        for iteration in range(self.n_iter):
            c_smooth, g_smooth = self._smoothness_cost_and_grad(xi)
            c_obs,    g_obs    = self._obstacle_cost_and_grad(xi)
            c_goal,   g_goal   = self._goal_cost_and_grad(xi, ee_goal)

            cost = (self.lambda_smooth * c_smooth
                    + c_obs
                    + lambda_goal * c_goal)
            grad = (self.lambda_smooth * g_smooth
                    + g_obs
                    + lambda_goal * g_goal)

            # --- Gradient clipping: bound the max step size ---------------
            grad_norm = np.linalg.norm(grad)
            max_grad  = 1.0 / self.learning_rate   # never move more than 1 rad/step
            if grad_norm > max_grad:
                grad = grad * (max_grad / grad_norm)

            xi -= self.learning_rate * grad

            # --- Project every waypoint to joint limits -------------------
            xi = np.clip(xi, self.joint_lb, self.joint_ub)

            # --- Fix start; end floats toward ee_goal ---------------------
            xi[0] = theta_start_fixed

            if abs(prev_cost - cost) < self.tol:
                break
            prev_cost = cost

        t3 = time.perf_counter()

        # ---- Re-fit B-spline to the optimized waypoints ------------------
        # Downsample optimized trajectory back to control points for a fair
        # comparison with the B-spline output of your planner.
        indices = np.linspace(0, self.num_samples - 1, self.num_control_points, dtype=int)
        new_control_points = xi[indices]
        bspline_out = BSpline(new_control_points)
        trajectory = bspline_out.spline_eval(self.num_samples)

        timings = {
            "initializer_rrt_time": t1 - t0,
            "solver_time": t3 - t2,
            "total_time": (t1 - t0) + (t3 - t2),
        }

        if return_timings:
            return trajectory, timings

        initial_guess = control_points
        return trajectory, initial_guess