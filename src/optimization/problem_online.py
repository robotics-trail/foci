from typing import Any, List

import casadi as cas
import numpy as np

from src.environment.convolution_online import ConvolutionFunctorWarpOnline
from src.splines.bspline import BSpline
from src.splines.minvo import minvo_hulls
from src.planning.joints import JointGroups


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    default = {
        "goal": 1.0,
        "obstacle": 1.0,
        "jerk": 1.0,
        "virtual_jerk": 1.0,
    }

    if weights is None:
        return default

    return {**default, **weights}


def _goal_cost(current_position, target_position, weight: float):
    return weight * cas.sum1((current_position - target_position) ** 2)


def _jerk_cost(
    dddcurve,
    duration,
    real_indices: List[int],
    virtual_indices: List[int],
    real_weight: float,
    virtual_weight: float,
):
    cost = 0.0
    duration_factor = duration**6

    if real_indices:
        real_jerk = dddcurve[:, real_indices]
        cost += real_weight * duration_factor * cas.sum1(cas.sum2(real_jerk**2)) / len(real_indices)

    if virtual_indices:
        virtual_jerk = dddcurve[:, virtual_indices]
        cost += virtual_weight * duration_factor * cas.sum1(cas.sum2(virtual_jerk**2)) / len(virtual_indices)

    return cost


def _flatten_covariances_row_major(covs_raw, n_gaussians: int):
    """Return row-major covariance vectors with shape (9*n_gaussians, 1).

    Supported symbolic layouts from robot.collision_covariances_online(q):
    - (9*n_gaussians, 1): already flattened row-major.
    - (n_gaussians, 9): one row-major covariance per Gaussian.
    - (3*n_gaussians, 3): one 3x3 block per Gaussian.

    CasADi matrices are 2D, so avoid returning a conceptual 3D object from
    robot.collision_covariances_online(q).
    """
    shape = covs_raw.shape

    if shape == (9 * n_gaussians, 1):
        return covs_raw

    if shape == (n_gaussians, 9):
        rows = []
        for g in range(n_gaussians):
            rows.extend([covs_raw[g, j] for j in range(9)])
        return cas.vertcat(*rows)

    if shape == (3 * n_gaussians, 3):
        rows = []
        for g in range(n_gaussians):
            cov = covs_raw[3 * g : 3 * g + 3, :]
            rows.extend(
                [
                    cov[0, 0], cov[0, 1], cov[0, 2],
                    cov[1, 0], cov[1, 1], cov[1, 2],
                    cov[2, 0], cov[2, 1], cov[2, 2],
                ]
            )
        return cas.vertcat(*rows)

    raise ValueError(
        "robot.collision_covariances_online(q) must return one of these shapes: "
        f"({9 * n_gaussians}, 1), ({n_gaussians}, 9), "
        f"or ({3 * n_gaussians}, 3). Got {shape}."
    )


def _obstacle_cost(
    collision_points,
    collision_covs,
    obstacle_means_param,
    obstacle_covs_param,
    n_gaussians: int,
    num_samples: int,
    weight: float,
):
    """Obstacle cost supporting per-sample (dynamic) obstacle positions.

    collision_points shape:
        (3*n_gaussians, num_samples)

    collision_covs shape:
        (9*n_gaussians, num_samples)

    obstacle_means_param shape:
        (num_samples, n_obstacles * 3)
        Row k contains the flattened means of all obstacles at sample k.

    obstacle_covs_param shape:
        (num_samples, n_obstacles * 9)
        Row k contains the flattened row-major covariances at sample k.
    """
    cost = 0
    callbacks = []

    n_obstacles = int(obstacle_means_param.shape[1]) // 3

    for sample_idx in range(num_samples):
        gaussian_points_flat = collision_points[:, sample_idx] 
        gaussian_points = cas.reshape(gaussian_points_flat, 3, n_gaussians).T 

        gaussian_covs_flat = collision_covs[:, sample_idx] 
        gaussian_covs = cas.reshape(gaussian_covs_flat, 9, n_gaussians).T 

        obstacle_points_flat = obstacle_means_param[sample_idx, :]
        obstacle_points = cas.reshape(obstacle_points_flat.T, 3, n_obstacles).T

        obstacle_covs_flat = obstacle_covs_param[sample_idx, :]
        obstacle_covs = cas.reshape(obstacle_covs_flat.T, 9, n_obstacles).T


        convolution = ConvolutionFunctorWarpOnline(
            f"conv_{sample_idx}",
            num_points=n_gaussians, 
            num_obstacles=n_obstacles
        )

        callbacks.append(convolution)
        cost += convolution(
            gaussian_points,
            gaussian_covs,
            obstacle_points,
            obstacle_covs,
        )

    return weight * cost / n_gaussians, callbacks


def _append_scalar_constraint(
    constraints,
    lower_bounds,
    upper_bounds,
    expr,
    lower: float,
    upper: float,
):
    constraints = cas.vertcat(constraints, expr)
    lower_bounds = np.concatenate((lower_bounds, [lower]))
    upper_bounds = np.concatenate((upper_bounds, [upper]))
    return constraints, lower_bounds, upper_bounds


def _build_constraints(
    symbolic_type,
    curve,
    start,
    n_dof: int,
    vel_hulls,
    acc_hulls,
    real_indices: List[int],
    virtual_indices: List[int],
    real_wmax: float,
    real_amax: float,
    virtual_wmax: float,
    virtual_amax: float,
):
    constraints = symbolic_type([])
    lbg = np.array([], dtype=float)
    ubg = np.array([], dtype=float)

    for i in range(n_dof):
        start_error = curve[0, i] - start[i]
        constraints, lbg, ubg = _append_scalar_constraint(
            constraints,
            lbg,
            ubg,
            start_error,
            0.0,
            0.0,
        )

    for hull in vel_hulls:
        for row in range(hull.shape[0]):
            virtual_joints_velocity_cost = 0.0
            for joint_idx in range(hull.shape[1]):
                if joint_idx in virtual_indices:
                    virtual_joints_velocity_cost += hull[row, joint_idx] ** 2
                else:
                    constraints, lbg, ubg = _append_scalar_constraint(
                        constraints,
                        lbg,
                        ubg,
                        hull[row, joint_idx],
                        -real_wmax,
                        real_wmax,
                    )

            if virtual_indices:
                constraints, lbg, ubg = _append_scalar_constraint(
                    constraints,
                    lbg,
                    ubg,
                    virtual_joints_velocity_cost,
                    0.0,
                    virtual_wmax**2,
                )

    for hull in acc_hulls:
        for row in range(hull.shape[0]):
            virtual_joints_acceleration_cost = 0.0
            for joint_idx in range(hull.shape[1]):
                if joint_idx in virtual_indices:
                    virtual_joints_acceleration_cost += hull[row, joint_idx] ** 2
                else:
                    constraints, lbg, ubg = _append_scalar_constraint(
                        constraints,
                        lbg,
                        ubg,
                        hull[row, joint_idx],
                        -real_amax,
                        real_amax,
                    )

            if virtual_indices:
                constraints, lbg, ubg = _append_scalar_constraint(
                    constraints,
                    lbg,
                    ubg,
                    virtual_joints_acceleration_cost,
                    0.0,
                    virtual_amax**2,
                )

    return constraints, lbg, ubg


def build_problem(
    robot,
    environment,
    num_control_points: int,
    num_samples: int,
    joint_groups: JointGroups,
    weights: dict[str, float] | None = None,
    vmax: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if num_control_points < 4:
        raise ValueError("num_control_points must be at least 4.")

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if vmax <= 0:
        raise ValueError("vmax must be positive.")

    weights = _normalize_weights(weights)

    symbolic_type = cas.MX
    n_dof = robot.n_dof

    joint_groups.validate(n_dof)
    real_indices = joint_groups.real_indices(n_dof)
    virtual_indices = joint_groups.virtual_indices

    control_points = symbolic_type.sym(
        "control_points",
        num_control_points,
        n_dof,
    )

    decision_variables = cas.vertcat(cas.vec(control_points))

    start = symbolic_type.sym("start", n_dof, 1)
    goal = symbolic_type.sym("goal", 3, 1)

    # Obstacle data is evaluated numerically at build time — not symbolic parameters.
    # Shape: (num_samples, n_obstacles * 3) and (num_samples, n_obstacles * 9)
    obstacle_means_num = environment.build_per_sample_means(num_samples)
    obstacle_covs_num = environment.build_per_sample_covariances(num_samples)

    obstacle_means_param = cas.DM(obstacle_means_num)
    obstacle_covs_param = cas.DM(obstacle_covs_num)

    params = cas.vertcat(
        cas.vec(start),
        cas.vec(goal),
    )

    bspline = BSpline(control_points)
    curve = bspline.spline_eval(num_samples)

    start_task = robot.f_task(curve[0, :])

    estimated_duration = cas.norm_2(goal - start_task) / vmax
    estimated_duration = cas.fmax(estimated_duration, 1e-3)

    num_segments = num_control_points - 4
    time_scale = num_segments / estimated_duration

    dddcurve = (time_scale**3) * bspline.spline_eval(
        num_samples,
        derivative_order=3,
    )

    q_sym = symbolic_type.sym("q", n_dof)

    collision_points_raw = robot.collision_points(q_sym)
    n_gaussians = int(collision_points_raw.shape[0])

    collision_covs_raw = robot.collision_covariances_online(q_sym)

    collision_points_vec = cas.reshape(
        collision_points_raw.T,
        3 * n_gaussians,
        1,
    )

    collision_covs_vec = _flatten_covariances_row_major(
        collision_covs_raw,
        n_gaussians,
    )

    collision_fun = cas.Function(
        "collision_points",
        [q_sym],
        [collision_points_vec],
    )
    collision_map = collision_fun.map(num_samples, "openmp")
    mapped_collision_points = collision_map(curve.T)

    covs_fun = cas.Function(
        "collision_covs",
        [q_sym],
        [collision_covs_vec],
    )
    covs_map = covs_fun.map(num_samples, "openmp")
    mapped_collision_covs = covs_map(curve.T)

    vel_hulls = [
        time_scale * h
        for h in minvo_hulls(control_points, derivative_order=1)
    ]

    acc_hulls = [
        (time_scale**2) * h
        for h in minvo_hulls(control_points, derivative_order=2)
    ]

    constraints, lbg, ubg = _build_constraints(
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

    final_task = robot.f_task(curve[-1, :])

    cost_goal = _goal_cost(
        final_task,
        goal,
        weight=weights["goal"],
    )

    cost_jerk = _jerk_cost(
        dddcurve,
        estimated_duration,
        real_indices=real_indices,
        virtual_indices=virtual_indices,
        real_weight=weights["jerk"],
        virtual_weight=weights["virtual_jerk"],
    )

    cost_obstacles, callbacks = _obstacle_cost(
        mapped_collision_points,
        mapped_collision_covs,
        obstacle_means_param,
        obstacle_covs_param,
        n_gaussians=n_gaussians,
        num_samples=num_samples,
        weight=weights["obstacle"],
    )

    total_cost = cost_goal + cost_obstacles + cost_jerk

    nlp = {
        "x": decision_variables,
        "f": total_cost,
        "p": params,
        "g": constraints,
    }

    return nlp, lbg, ubg, callbacks