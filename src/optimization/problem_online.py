"""Online trajectory optimisation problem builder.

Obstacle statistics are queried per sample index k from the environment, so
mobile obstacles are correctly accounted for.  Robot covariances may depend
on configuration q (symbolic) and are mapped over the trajectory as well.
"""

from __future__ import annotations

from typing import Any

import casadi as cas
import numpy as np

from src.environment.convolution import ConvolutionFunctorOnline
from src.splines.bspline import BSpline
from src.splines.minvo import minvo_hulls
from src.planning.joints import JointGroups

from src.optimization.utils import (
    normalize_weights,
    goal_cost,
    jerk_cost,
    build_constraints,
    build_collision_point_map,
    estimate_duration,
    validate_build_inputs,
)


# ---------------------------------------------------------------------------
# Online-specific helpers
# ---------------------------------------------------------------------------


def _flatten_covariances_row_major(covs_raw, n_gaussians: int):
    """Normalise the symbolic covariance output of robot.collision_covariances_online.

    Accepts three CasADi matrix layouts and returns a column vector of shape
    (9 * n_gaussians, 1) in row-major order.

    Supported input shapes
    ----------------------
    (9*n_gaussians, 1)  — already flattened.
    (n_gaussians, 9)    — one row-major covariance per Gaussian.
    (3*n_gaussians, 3)  — one 3×3 block per Gaussian (stacked vertically).
    """
    shape = covs_raw.shape

    if shape == (9 * n_gaussians, 1):
        return covs_raw

    if shape == (n_gaussians, 9):
        return cas.vertcat(*[covs_raw[g, j] for g in range(n_gaussians) for j in range(9)])

    if shape == (3 * n_gaussians, 3):
        rows = []
        for g in range(n_gaussians):
            cov = covs_raw[3 * g : 3 * g + 3, :]
            rows.extend([
                cov[0, 0], cov[0, 1], cov[0, 2],
                cov[1, 0], cov[1, 1], cov[1, 2],
                cov[2, 0], cov[2, 1], cov[2, 2],
            ])
        return cas.vertcat(*rows)

    raise ValueError(
        "robot.collision_covariances_online(q) must return one of: "
        f"({9 * n_gaussians}, 1), ({n_gaussians}, 9), ({3 * n_gaussians}, 3). "
        f"Got {shape}."
    )


def _build_collision_cov_map(robot, q_sym, n_gaussians: int, num_samples: int, curve):
    """Map robot.collision_covariances_online over all samples in parallel.

    Returns
    -------
    mapped_collision_covs : shape (9*n_gaussians, num_samples)
    """
    covs_raw = robot.collision_covariances_online(q_sym)
    covs_vec = _flatten_covariances_row_major(covs_raw, n_gaussians)

    covs_fun = cas.Function("collision_covs", [q_sym], [covs_vec])
    covs_map = covs_fun.map(num_samples, "openmp")
    return covs_map(curve.T)                                   # (9*n_gaussians, num_samples)


def _obstacle_cost(
    collision_points,
    collision_covs,
    obstacle_means_param,
    obstacle_covs_param,
    n_gaussians: int,
    num_samples: int,
    weight: float,
):
    """Obstacle cost for an online (dynamic) environment.

    One ConvolutionFunctorOnline is instantiated per sample; all four
    tensors (robot means, robot covs, obstacle means, obstacle covs) are
    passed at eval-time.

    Parameters
    ----------
    collision_points:
        Shape (3*n_gaussians, num_samples).
    collision_covs:
        Shape (9*n_gaussians, num_samples).
    obstacle_means_param:
        cas.DM, shape (num_samples, n_obstacles * 3).
    obstacle_covs_param:
        cas.DM, shape (num_samples, n_obstacles * 9).
    """
    n_obstacles = int(obstacle_means_param.shape[1]) // 3

    cost = 0
    callbacks = []

    for k in range(num_samples):
        # Robot quantities at sample k — reshape to (n_gaussians, 3/9)
        gaussian_points = cas.reshape(collision_points[:, k], 3, n_gaussians).T
        gaussian_covs   = cas.reshape(collision_covs[:,   k], 9, n_gaussians).T

        # Obstacle quantities at sample k — reshape to (n_obstacles, 3/9)
        obstacle_points = cas.reshape(obstacle_means_param[k, :].T, 3, n_obstacles).T
        obstacle_covs   = cas.reshape(obstacle_covs_param[k,  :].T, 9, n_obstacles).T

        convolution = ConvolutionFunctorOnline(
            f"conv_{k}",
            num_points=n_gaussians,
            num_obstacles=n_obstacles,
        )
        callbacks.append(convolution)
        cost += convolution(gaussian_points, gaussian_covs, obstacle_points, obstacle_covs)

    return weight * cost / n_gaussians, callbacks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_problem(
    robot,
    environment,
    num_control_points: int,
    num_samples: int,
    joint_groups: JointGroups,
    weights: dict[str, float] | None = None,
    vmax: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list]:
    """Build the online trajectory optimisation NLP.

    Returns
    -------
    nlp, lbg, ubg, callbacks
    """
    validate_build_inputs(num_control_points, num_samples, vmax)

    weights = normalize_weights(weights)

    symbolic_type = cas.MX
    n_dof = robot.n_dof

    joint_groups.validate(n_dof)
    real_indices    = joint_groups.real_indices(n_dof)
    virtual_indices = joint_groups.virtual_indices

    # --- Decision variables and parameters ---------------------------
    control_points     = symbolic_type.sym("control_points", num_control_points, n_dof)
    decision_variables = cas.vertcat(cas.vec(control_points))

    start  = symbolic_type.sym("start", n_dof, 1)
    goal   = symbolic_type.sym("goal",  3,     1)
    params = cas.vertcat(cas.vec(start), cas.vec(goal))

    # --- Obstacle data baked in as CasADi constants ------------------
    # Queried per sample k at build time; not re-evaluated during solve.
    obstacle_means_param = cas.DM(environment.build_per_sample_means(num_samples))
    obstacle_covs_param  = cas.DM(environment.build_per_sample_covariances(num_samples))

    # --- Spline and time scaling -------------------------------------
    bspline = BSpline(control_points)
    curve   = bspline.spline_eval(num_samples)

    start_task         = robot.f_task(curve[0, :])
    estimated_duration = estimate_duration(goal, start_task, vmax)
    time_scale         = (num_control_points - 4) / estimated_duration

    dddcurve = time_scale ** 3 * bspline.spline_eval(num_samples, derivative_order=3)

    # --- Collision points and covariances ----------------------------
    q_sym = symbolic_type.sym("q", n_dof)
    n_gaussians = int(robot.collision_points(q_sym).shape[0])

    mapped_collision_points = build_collision_point_map(
        robot, q_sym, n_gaussians, num_samples, curve
    )
    mapped_collision_covs = _build_collision_cov_map(
        robot, q_sym, n_gaussians, num_samples, curve
    )

    # --- Constraints -------------------------------------------------
    vel_hulls = [time_scale      * h for h in minvo_hulls(control_points, derivative_order=1)]
    acc_hulls = [time_scale ** 2 * h for h in minvo_hulls(control_points, derivative_order=2)]

    constraints, lbg, ubg = build_constraints(
        symbolic_type=symbolic_type,
        curve=curve,
        start=start,
        n_dof=n_dof,
        vel_hulls=vel_hulls,
        acc_hulls=acc_hulls,
        real_indices=real_indices,
        virtual_indices=virtual_indices,
        real_wmax=joint_groups.real_wmax,
        real_amax=joint_groups.real_amax,
        virtual_wmax=joint_groups.virtual_wmax,
        virtual_amax=joint_groups.virtual_amax,
    )

    # --- Costs -------------------------------------------------------
    cost_goal = goal_cost(robot.f_task(curve[-1, :]), goal, weight=weights["goal"])

    cost_jerk = jerk_cost(
        dddcurve, estimated_duration,
        real_indices=real_indices, virtual_indices=virtual_indices,
        real_weight=weights["jerk"], virtual_weight=weights["virtual_jerk"],
    )

    cost_obstacles, callbacks = _obstacle_cost(
        mapped_collision_points, mapped_collision_covs,
        obstacle_means_param, obstacle_covs_param,
        n_gaussians=n_gaussians, num_samples=num_samples, weight=weights["obstacle"],
    )

    # --- NLP ---------------------------------------------------------
    nlp = {
        "x": decision_variables,
        "f": cost_goal + cost_obstacles + cost_jerk,
        "p": params,
        "g": constraints,
    }

    return nlp, lbg, ubg, callbacks