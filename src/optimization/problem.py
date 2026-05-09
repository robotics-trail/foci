from typing import Any, List

import casadi as cas
import numpy as np

from src.environment.convolution import ConvolutionFunctorWarp
from src.splines.bspline import BSpline
from src.splines.minvo import minvo_hulls
from src.planning.joints import JointGroups


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    default = {
        "goal": 1.0,
        "obstacle": 1.0,
        "jerk": 1.0,
        "virtual_jerk": 1.0
    }

    if weights is None:
        return default

    return {**default, **weights}


def _goal_cost(
    current_position,
    target_position,
    weight: float,
):
    return weight * cas.sum1((current_position - target_position) ** 2)


def _jerk_cost(
    dddcurve,
    real_indices: List[int], 
    virtual_indices: List[int], 
    real_weight: float,
    virtual_weight: float
):
    
    cost: float = 0.0

    if real_indices: 
        real_jerk = dddcurve[:, real_indices]
        cost += real_weight * cas.sum1(cas.sum2(real_jerk**2))

    if virtual_indices: 
        virtual_jerk = dddcurve[:, virtual_indices]
        cost += virtual_weight * cas.sum1(cas.sum2(virtual_jerk**2))
    
    return cost


def _obstacle_cost(
    collision_points,
    environment,
    robot_covariances: np.ndarray,
    num_samples: int,
    weight: float,
):
    """
    Obstacle cost with one covariance per robot Gaussian.

    collision_points has shape:

        (num_samples * n_gaussians, 3)

    robot_covariances has shape:

        (n_gaussians, 3, 3)
    """
    robot_covariances = np.asarray(robot_covariances, dtype=float)

    if robot_covariances.ndim != 3 or robot_covariances.shape[1:] != (3, 3):
        raise ValueError(
            "robot_covariances must have shape (n_gaussians, 3, 3)."
        )

    n_gaussians = robot_covariances.shape[0]

    cost = 0
    callbacks = []

    for gaussian_idx in range(n_gaussians):
        gaussian_points = collision_points[
            gaussian_idx::n_gaussians,
            :
        ]

        covs = environment.obstacle_covariances + robot_covariances[gaussian_idx]
        covs_det = np.linalg.det(covs)
        covs_inv = np.linalg.inv(covs)

        convolution = ConvolutionFunctorWarp(
            f"conv_robot_gaussian_{gaussian_idx}",
            3,
            num_samples,
            environment.obstacle_means,
            covs_det,
            covs_inv,
        )

        callbacks.append(convolution)
        cost += convolution(gaussian_points)

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
    real_wmax: float,
    real_amax: float,
    virtual_wmax: float,
    virtual_amax: float,
):
    constraints = symbolic_type([])
    lbg = np.array([], dtype=float)
    ubg = np.array([], dtype=float)

    # Start equality constraint.
    start_error = sum(
        (curve[0, i] - start[i]) ** 2
        for i in range(n_dof)
    )

    constraints, lbg, ubg = _append_scalar_constraint(
        constraints,
        lbg,
        ubg,
        start_error,
        0.0,
        0.0,
    )

    # Velocity hull component-wise bounds.
    for hull in vel_hulls:
        for row in range(hull.shape[0]):
            for joint_idx in range(hull.shape[1]):
                if joint_idx in real_indices:
                    constraints, lbg, ubg = _append_scalar_constraint(
                        constraints,
                        lbg,
                        ubg,
                        hull[row, joint_idx],
                        -real_wmax,
                        real_wmax,
                    )

                else: 
                    constraints, lbg, ubg = _append_scalar_constraint(
                        constraints,
                        lbg,
                        ubg,
                        hull[row, joint_idx],
                        -virtual_wmax,
                        virtual_wmax,
                    )

    # Acceleration hull component-wise bounds.
    for hull in acc_hulls:
        for row in range(hull.shape[0]):
            for joint_idx in range(hull.shape[1]):
                if joint_idx in real_indices:
                    constraints, lbg, ubg = _append_scalar_constraint(
                        constraints,
                        lbg,
                        ubg,
                        hull[row, joint_idx],
                        -real_amax,
                        real_amax,
                    )

                else: 
                    constraints, lbg, ubg = _append_scalar_constraint(
                        constraints,
                        lbg,
                        ubg,
                        hull[row, joint_idx],
                        -virtual_amax,
                        virtual_amax,
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
    """
    Build the trajectory optimization NLP.

    Parameters
    ----------
    robot:
        Any robot implementing:

            robot.n_dof
            robot.f_task(q)
            robot.collision_points(q)

    environment:
        GaussianEnvironment.

    robot_covariance:
        3x3 covariance matrix for the robot collision model.
        If omitted, a small isotropic covariance is used.

    Returns
    -------
    nlp, lbg, ubg
    """
    if num_control_points < 4:
        raise ValueError("num_control_points must be at least 4.")

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if vmax <= 0:
        raise ValueError("vmax must be positive.")

    weights = _normalize_weights(weights)

    robot_covariance = robot.collision_covariances()

    symbolic_type = cas.MX
    n_dof = robot.n_dof

    joint_groups.validate(n_dof)
    real_indices = joint_groups.real_indices(n_dof)
    virtual_indices = joint_groups.virtual_indices

    # ==========================================================
    # Decision variables
    # ==========================================================

    control_points = symbolic_type.sym(
        "control_points",
        num_control_points,
        n_dof,
    )

    decision_variables = cas.vertcat(cas.vec(control_points))

    # ==========================================================
    # Parameters
    # ==========================================================

    start = symbolic_type.sym("start", n_dof, 1)
    goal = symbolic_type.sym("goal", 3, 1)

    params = cas.vertcat(
        cas.vec(start),
        cas.vec(goal),
    )

    # ==========================================================
    # Spline
    # ==========================================================

    bspline = BSpline(control_points)

    curve = bspline.spline_eval(num_samples)

    start_task = robot.f_task(curve[0, :])

    estimated_duration = cas.norm_2(goal - start_task) / vmax

    # Avoid division by zero in very small motions.
    estimated_duration = cas.fmax(estimated_duration, 1e-3)

    num_segments = num_control_points - 4
    time_scale = num_segments / estimated_duration

    dcurve = time_scale * bspline.spline_eval(
        num_samples,
        derivative_order=1,
    )

    ddcurve = (time_scale**2) * bspline.spline_eval(
        num_samples,
        derivative_order=2,
    )

    dddcurve = (time_scale**3) * bspline.spline_eval(
        num_samples,
        derivative_order=3,
    )

    # ==========================================================
    # Collision points
    # ==========================================================

    q_sym = symbolic_type.sym("q", n_dof)

    collision_fun = cas.Function(
        "collision_points",
        [q_sym],
        [robot.collision_points(q_sym)],
    )
    print(curve.shape)

    collision_map = collision_fun.map(num_samples, "openmp")

    # mapped_collision_points: (3*n_gaussians, num_samples)
    mapped_collision_points = collision_map(curve.T)

    n_gaussians = len(robot.gaussian_specs)

    # Cada columna de mapped_collision_points es un sample con layout [x0,y0,z0,x1,y1,z1,...]
    # Queremos (num_samples * n_gaussians, 3) con orden [s0_g0, s0_g1, ..., s1_g0, ...]
    # reshape column-major de CasADi: primero varía la fila, luego la columna
    # (3*n_gaussians, num_samples) -> leer columna a columna -> (3, n_gaussians*num_samples) -> .T
    collision_points = cas.reshape(
        mapped_collision_points,
        3,
        n_gaussians * num_samples,
    ).T

    # ==========================================================
    # Constraints
    # ==========================================================

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
        real_wmax=joint_groups.real_wmax,
        real_amax=joint_groups.real_amax,
        virtual_wmax=joint_groups.virtual_wmax, 
        virtual_amax=joint_groups.virtual_amax
    )

    # ==========================================================
    # Costs
    # ==========================================================

    final_task = robot.f_task(curve[-1, :])

    cost_goal = _goal_cost(
        final_task,
        goal,
        weight=weights["goal"],
    )

    cost_jerk = _jerk_cost(
        dddcurve,
        real_indices=real_indices, 
        virtual_indices=virtual_indices, 
        real_weight=weights["jerk"], 
        virtual_weight=weights["virtual_jerk"]
    )

    cost_obstacles, callbacks = _obstacle_cost(
        collision_points,
        environment,
        robot_covariance,
        num_samples=num_samples,
        weight=weights["obstacle"],
    )

    total_cost = cost_goal + cost_obstacles + cost_jerk

    # ==========================================================
    # NLP
    # ==========================================================

    nlp = {
        "x": decision_variables,
        "f": total_cost,
        "p": params,
        "g": constraints,
    }

    return nlp, lbg, ubg, callbacks