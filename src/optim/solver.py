"""
Utilities to build CasADi/IPOPT trajectory optimization solvers.

This module provides:
- a shared NLP construction pipeline
- collision-point builders for different robot obstacle models
- obstacle-cost builders for midpoint and multi-Gaussian formulations
- public solver factories for the two supported planning modes
"""

from typing import Callable, Dict, List, Optional, Tuple

import casadi as cas
import numpy as np

from src.core.convolution import ConvolutionFunctorWarp
from src.core.robot_loader import ManipulatorRobotURDF
from src.optim.constraints import build_constraints
from src.optim.cost_functions import get_cost_goal, get_cost_obstacles, get_cost_jerk
from src.splines.bspline import BSpline
from src.splines.minvo import minvo_hulls


DEFAULT_WEIGHTS = {
    "jerk": 1.0,
    "goal": 1.0,
    "obstacle": 1.0,
}


def _normalize_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Fill missing optimization weights with default values.

    Parameters
    ----------
    weights : dict or None
        Optional dictionary with keys ``jerk``, ``goal``, and ``obstacle``.

    Returns
    -------
    dict
        Complete weights dictionary.
    """
    if weights is None:
        return DEFAULT_WEIGHTS.copy()

    return {**DEFAULT_WEIGHTS, **weights}


def _validate_solver_inputs(
    num_control_points: int,
    num_samples: int,
    vmax: float,
) -> None:
    """
    Validate common public solver inputs.

    Raises
    ------
    ValueError
        If any required input is invalid.
    """
    if num_control_points < 4:
        raise ValueError(
            f"num_control_points must be at least 4 for a cubic B-spline, got {num_control_points}."
        )

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    if vmax <= 0:
        raise ValueError(f"vmax must be positive, got {vmax}.")


def _get_ipopt_options() -> Dict[str, object]:
    """
    Return IPOPT options used for the trajectory optimization problem.
    """
    return {
        "ipopt.print_level": 5,
        "ipopt.max_iter": 1_000,
        "ipopt.tol": 1e-3,
        "print_time": 0,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_obj_change_tol": 1e-3,
        "ipopt.constr_viol_tol": 1e-3,
        "ipopt.acceptable_iter": 3,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
    }


def _create_solver_common(
    robot: ManipulatorRobotURDF,
    num_control_points: int,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians: bool,
    active_link_indices: List[int],
    num_samples: int,
    weights: Dict[str, float],
    wmax: float,
    vmax: float,
    amax: float,
    point_builder: Callable,
    obstacle_cost_builder: Callable,
    solver_name: str,
    fk_name: str = "fk",
):
    """
    Build a trajectory-optimization NLP using a shared construction pipeline.

    This function assembles:
    - spline decision variables
    - start/goal parameters
    - time-scaled spline derivatives
    - forward kinematics along the trajectory
    - collision evaluation points via ``point_builder``
    - velocity and acceleration hull constraints
    - goal, obstacle, and jerk costs
    - the final IPOPT solver

    Parameters
    ----------
    robot : ManipulatorRobotURDF
        Robot model used for forward kinematics.
    num_control_points : int
        Number of spline control points.
    obstacle_means : np.ndarray
        Obstacle Gaussian means.
    covs_det, covs_inv :
        Precomputed covariance determinants and inverses.
    multiple_gaussians : bool
        Whether link-dependent covariance models are used.
    active_link_indices : list[int]
        Links considered in obstacle evaluation.
    num_samples : int
        Number of spline samples used in the optimization.
    weights : dict
        Optimization weights with keys ``goal``, ``obstacle``, and ``jerk``.
    wmax, vmax, amax : float
        Motion limits used in scaling and constraints.
    point_builder : callable
        Function that builds collision evaluation points.
    obstacle_cost_builder : callable
        Function that builds the obstacle term and its convolution functor(s).
    solver_name : str
        Name of the CasADi NLP solver instance.
    fk_name : str, default="fk"
        Name of the internal FK CasADi function.

    Returns
    -------
    tuple
        ``(solver, lbg, ubg, convolution_functor)``.
    """
    symbolic_type = cas.MX
    n_joints = robot.get_n_joints()
    n_links = robot.get_n_links()

    # --- Decision variables ---
    control_points = symbolic_type.sym("control_points", num_control_points, n_joints)
    decision_variables = cas.vertcat(cas.vec(control_points))

    # --- Parameters ---
    start_conf = symbolic_type.sym("start_conf", n_joints, 1)
    goal_ee_position = symbolic_type.sym("goal_ee_position", 3, 1)
    params = cas.vertcat(cas.vec(start_conf), cas.vec(goal_ee_position))

    # --- Spline evaluation ---
    bspline = BSpline(control_points)
    curve = bspline.spline_eval(num_samples)

    start_ee_position = robot.get_ee_endpoint(curve[0, :])

    estimated_duration = cas.norm_2(goal_ee_position - start_ee_position) / vmax
    num_segments = num_control_points - 4
    time_to_spline_scale = num_segments / estimated_duration

    dcurve = time_to_spline_scale * bspline.spline_eval(num_samples, derivative_order=1)
    ddcurve = (time_to_spline_scale**2) * bspline.spline_eval(
        num_samples, derivative_order=2
    )
    dddcurve = (time_to_spline_scale**3) * bspline.spline_eval(
        num_samples, derivative_order=3
    )

    print(dddcurve)

    # --- Kinematics ---
    q_sym = symbolic_type.sym("q", n_joints)
    fk_function = cas.Function(fk_name, [q_sym], [robot.forward_kinematics(q_sym)])
    kinematics_functor = fk_function.map(num_samples, "openmp")

    joint_positions = kinematics_functor(curve.T).T
    joint_positions_reshaped = cas.reshape(
        joint_positions,
        num_samples * (n_links + 1),
        3,
    )

    # --- Collision evaluation points ---
    collision_points, aux_data = point_builder(
        symbolic_type=symbolic_type,
        joint_positions_reshaped=joint_positions_reshaped,
        num_samples=num_samples,
        n_links=n_links,
        active_link_indices=active_link_indices,
    )

    # --- Hulls ---
    vel_hulls = [
        time_to_spline_scale * h
        for h in minvo_hulls(control_points, derivative_order=1)
    ]
    acc_hulls = [
        (time_to_spline_scale**2) * h
        for h in minvo_hulls(control_points, derivative_order=2)
    ]

    # --- Constraints ---
    constraints, lbg, ubg = build_constraints(
        symbolic_type,
        curve,
        start_conf,
        n_joints,
        vel_hulls,
        acc_hulls,
        start_limit=0.0,
        vel_limit=wmax**2,
        acc_limit=amax**2,
    )

    # --- Costs ---
    final_ee_pos = robot.get_ee_endpoint(curve[-1, :])

    cost_goal = get_cost_goal(
        final_ee_pos,
        goal_ee_position,
        weight=weights["goal"],
    )

    cost_jerk = get_cost_jerk(
        dddcurve,
        weight=weights["jerk"],
    )

    cost_obstacles, convolution_functor = obstacle_cost_builder(
        collision_points=collision_points,
        aux_data=aux_data,
        obstacle_means=obstacle_means,
        covs_det=covs_det,
        covs_inv=covs_inv,
        multiple_gaussians=multiple_gaussians,
        active_link_indices=active_link_indices,
        num_samples=num_samples,
        obstacle_weight=weights["obstacle"],
    )

    total_cost = cost_goal + cost_obstacles + cost_jerk

    # --- NLP ---
    nlp = {
        "x": decision_variables,
        "f": total_cost,
        "p": params,
        "g": constraints,
    }

    solver = cas.nlpsol(solver_name, "ipopt", nlp, _get_ipopt_options())
    return solver, lbg, ubg, convolution_functor


def _build_midpoints(
    symbolic_type,
    joint_positions_reshaped,
    num_samples: int,
    n_links: int,
    active_link_indices: List[int],
):
    """
    Build one collision evaluation point per active link and spline sample.

    Each point is the midpoint of the corresponding robot link segment.

    Returns
    -------
    tuple
        ``(midpoints, None)`` where ``midpoints`` has shape
        ``(num_samples * n_active_links, 3)``.
    """
    n_active_links = len(active_link_indices)
    midpoints = symbolic_type.zeros(num_samples * n_active_links, 3)

    for sample_idx in range(num_samples):
        base_idx = sample_idx * (n_links + 1)

        for local_idx, link_idx in enumerate(active_link_indices):
            joint1_idx = base_idx + link_idx
            joint2_idx = base_idx + link_idx + 1

            midpoint_idx = sample_idx * n_active_links + local_idx
            midpoints[midpoint_idx, :] = (
                joint_positions_reshaped[joint1_idx, :]
                + joint_positions_reshaped[joint2_idx, :]
            ) / 2.0

    return midpoints, None


def _build_gaussian_points(
    symbolic_type,
    joint_positions_reshaped,
    num_samples: int,
    n_links: int,
    active_link_indices: List[int],
    gaussian_specs: List[Tuple[int, float]],
):
    """
    Build arbitrary Gaussian evaluation points along robot links.

    Parameters
    ----------
    gaussian_specs : list[tuple[int, float]]
        List of ``(link_idx, t)`` pairs, where ``t in [0, 1]`` interpolates
        along the segment joining consecutive link positions.

    Returns
    -------
    tuple
        ``(gaussian_points, aux_data)`` where:
        - ``gaussian_points`` has shape ``(num_samples * n_total_gaussians, 3)``
        - ``aux_data["points_by_link"]`` groups points by link index
        - ``aux_data["gaussian_specs"]`` stores the original point definitions
    """
    n_total_gaussians = len(gaussian_specs)
    gaussian_points = symbolic_type.zeros(num_samples * n_total_gaussians, 3)
    points_by_link = {link_idx: [] for link_idx in active_link_indices}

    for sample_idx in range(num_samples):
        base_joint_idx = sample_idx * (n_links + 1)

        for gaussian_idx, (link_idx, t) in enumerate(gaussian_specs):
            p0 = joint_positions_reshaped[base_joint_idx + link_idx, :]
            p1 = joint_positions_reshaped[base_joint_idx + link_idx + 1, :]

            point = (1.0 - t) * p0 + t * p1

            flat_idx = sample_idx * n_total_gaussians + gaussian_idx
            gaussian_points[flat_idx, :] = point
            points_by_link[link_idx].append(point)

    aux_data = {
        "points_by_link": points_by_link,
        "gaussian_specs": gaussian_specs,
    }
    return gaussian_points, aux_data


def _build_obstacle_cost_midpoints(
    collision_points,
    aux_data,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians: bool,
    active_link_indices: List[int],
    num_samples: int,
    obstacle_weight: float,
):
    """
    Build obstacle cost for the midpoint-based planner.

    If ``multiple_gaussians`` is False, a single convolution functor is used
    for all midpoint samples. Otherwise, one functor per active link is created.

    Returns
    -------
    tuple
        ``(cost_obstacles, convolution_functor)``.
    """
    del aux_data  # unused in this builder
    n_active_links = len(active_link_indices)

    if not multiple_gaussians:
        convolution_functor = ConvolutionFunctorWarp(
            "conv",
            3,
            n_active_links * num_samples,
            obstacle_means,
            covs_det,
            covs_inv,
        )
    else:
        convolution_functor = []
        for local_idx in range(n_active_links):
            conv_func = ConvolutionFunctorWarp(
                f"conv_link_{local_idx}",
                3,
                num_samples,
                obstacle_means,
                covs_det[local_idx],
                covs_inv[local_idx],
            )
            convolution_functor.append(conv_func)

    cost_obstacles = get_cost_obstacles(
        collision_points,
        convolution_functor,
        multiple_gaussians,
        n_active_links,
        weight=obstacle_weight,
    )

    return cost_obstacles, convolution_functor


def _build_obstacle_cost_multi_gauss(
    collision_points,
    aux_data,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians: bool,
    active_link_indices: List[int],
    num_samples: int,
    obstacle_weight: float,
):
    """
    Build obstacle cost for the multi-Gaussian planner.

    In the shared-covariance case, a single convolution functor is evaluated
    over all Gaussian points. In the link-dependent covariance case, one
    convolution functor per active link is used and the final result is
    averaged over the total number of points.

    Returns
    -------
    tuple
        ``(cost_obstacles, convolution_functor)``.
    """
    del num_samples  # unused in this builder
    points_by_link = aux_data["points_by_link"]
    n_total_points = collision_points.shape[0]

    if not multiple_gaussians:
        convolution_functor = ConvolutionFunctorWarp(
            "conv_multi_gauss",
            3,
            n_total_points,
            obstacle_means,
            covs_det,
            covs_inv,
        )
        cost_obstacles = obstacle_weight * convolution_functor(collision_points)
        return cost_obstacles, convolution_functor

    active_link_to_local = {
        link_idx: local_idx for local_idx, link_idx in enumerate(active_link_indices)
    }

    convolution_functor = []
    weighted_sum = 0
    total_points = 0

    for link_idx in active_link_indices:
        link_points_list = points_by_link[link_idx]
        n_link_points = len(link_points_list)

        if n_link_points == 0:
            continue

        local_idx = active_link_to_local[link_idx]
        link_points_mat = cas.vertcat(*link_points_list)

        conv_func = ConvolutionFunctorWarp(
            f"conv_multi_gauss_link_{link_idx}",
            3,
            n_link_points,
            obstacle_means,
            covs_det[local_idx],
            covs_inv[local_idx],
        )
        convolution_functor.append(conv_func)

        weighted_sum += n_link_points * conv_func(link_points_mat)
        total_points += n_link_points

    if total_points == 0:
        raise ValueError("No Gaussian points assigned to active links.")

    cost_obstacles = obstacle_weight * (weighted_sum / total_points)
    return cost_obstacles, convolution_functor


def create_solver(
    robot: ManipulatorRobotURDF,
    num_control_points: int,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians: bool,
    active_link_indices: List[int],
    num_samples: int = 30,
    weights: Optional[Dict[str, float]] = None,
    wmax: float = 1.0,
    vmax: float = 1.0,
    amax: float = 1.0,
):
    """
    Create the standard midpoint-based trajectory optimization solver.

    Parameters
    ----------
    robot : ManipulatorRobotURDF
        Robot model.
    num_control_points : int
        Number of spline control points.
    obstacle_means : np.ndarray
        Obstacle Gaussian means.
    covs_det, covs_inv :
        Precomputed obstacle covariance determinants and inverses.
    multiple_gaussians : bool
        Whether link-dependent covariance models are used.
    active_link_indices : list[int]
        Links considered in obstacle evaluation.
    num_samples : int, default=30
        Number of spline samples used in the optimization.
    weights : dict or None, default=None
        Optimization weights. Missing keys are filled with defaults.
    wmax, vmax, amax : float, default=1.0
        Motion limits used in scaling and constraints.

    Returns
    -------
    tuple
        ``(solver, lbg, ubg, convolution_functor)``.
    """
    _validate_solver_inputs(num_control_points, num_samples, vmax)
    weights = _normalize_weights(weights)

    return _create_solver_common(
        robot=robot,
        num_control_points=num_control_points,
        obstacle_means=obstacle_means,
        covs_det=covs_det,
        covs_inv=covs_inv,
        multiple_gaussians=multiple_gaussians,
        active_link_indices=active_link_indices,
        num_samples=num_samples,
        weights=weights,
        wmax=wmax,
        vmax=vmax,
        amax=amax,
        point_builder=_build_midpoints,
        obstacle_cost_builder=_build_obstacle_cost_midpoints,
        solver_name="solver",
        fk_name="fk",
    )


def create_multiple_gaussians_solver(
    robot: ManipulatorRobotURDF,
    num_control_points: int,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians: bool,
    active_link_indices: List[int],
    gaussian_specs: List[Tuple[int, float]],
    num_samples: int = 30,
    weights: Optional[Dict[str, float]] = None,
    wmax: float = 1.0,
    vmax: float = 1.0,
    amax: float = 1.0,
):
    """
    Create the multi-Gaussian trajectory optimization solver.

    Parameters
    ----------
    robot : ManipulatorRobotURDF
        Robot model.
    num_control_points : int
        Number of spline control points.
    obstacle_means : np.ndarray
        Obstacle Gaussian means.
    covs_det, covs_inv :
        Precomputed obstacle covariance determinants and inverses.
    multiple_gaussians : bool
        Whether link-dependent covariance models are used.
    active_link_indices : list[int]
        Links considered in obstacle evaluation.
    gaussian_specs : list[tuple[int, float]]
        List of Gaussian sampling points along links as ``(link_idx, t)``.
    num_samples : int, default=30
        Number of spline samples used in the optimization.
    weights : dict or None, default=None
        Optimization weights. Missing keys are filled with defaults.
    wmax, vmax, amax : float, default=1.0
        Motion limits used in scaling and constraints.

    Returns
    -------
    tuple
        ``(solver, lbg, ubg, convolution_functor)``.
    """
    _validate_solver_inputs(num_control_points, num_samples, vmax)
    weights = _normalize_weights(weights)

    if len(gaussian_specs) == 0:
        raise ValueError("gaussian_specs must contain at least one Gaussian point.")

    def point_builder(**kwargs):
        return _build_gaussian_points(
            **kwargs,
            gaussian_specs=gaussian_specs,
        )

    return _create_solver_common(
        robot=robot,
        num_control_points=num_control_points,
        obstacle_means=obstacle_means,
        covs_det=covs_det,
        covs_inv=covs_inv,
        multiple_gaussians=multiple_gaussians,
        active_link_indices=active_link_indices,
        num_samples=num_samples,
        weights=weights,
        wmax=wmax,
        vmax=vmax,
        amax=amax,
        point_builder=point_builder,
        obstacle_cost_builder=_build_obstacle_cost_multi_gauss,
        solver_name="solver_multi_gauss",
        fk_name="fk_multi_gauss",
    )


def create_drone_solver(
    robot,
    num_control_points: int,
    obstacle_means,
    covs_det,
    covs_inv,
    num_samples: int = 30,
    weights: Optional[Dict[str, float]] = None,
    wmax: float = 1.0,
    vmax: float = 1.0,
    amax: float = 1.0,
):

    symbolic_type = cas.MX
    n_joints = robot.get_n_joints()

    # --- Decision variables ---
    control_points = symbolic_type.sym("control_points", num_control_points, n_joints)
    decision_variables = cas.vertcat(cas.vec(control_points))

    # --- Parameters ---
    start_conf = symbolic_type.sym("start_conf", n_joints, 1)
    goal_position = symbolic_type.sym("goal_position", 3, 1)
    params = cas.vertcat(cas.vec(start_conf), cas.vec(goal_position))

    # --- Spline evaluation ---
    bspline = BSpline(control_points)
    curve = bspline.spline_eval(num_samples)

    start_pos = robot.get_goal_point(curve[0, :])
    estimated_duration = cas.norm_2(goal_position - start_pos) / vmax
    num_segments = num_control_points - 4
    time_to_spline_scale = num_segments / estimated_duration

    dcurve = time_to_spline_scale * bspline.spline_eval(num_samples, derivative_order=1)
    ddcurve = (time_to_spline_scale**2) * bspline.spline_eval(
        num_samples, derivative_order=2
    )
    dddcurve = (time_to_spline_scale**3) * bspline.spline_eval(
        num_samples, derivative_order=3
    )

    # --- Collision points ---
    q_sym = symbolic_type.sym("q", n_joints)
    collision_fun = cas.Function(
        "drone_collision_points",
        [q_sym],
        [robot.get_collision_points(q_sym)],
    )
    collision_map = collision_fun.map(num_samples, "openmp")

    collision_points = collision_map(curve.T)

    # reshape to (num_samples * 3, 3)
    collision_points = cas.reshape(collision_points.T, num_samples * 3, 3)

    # --- Constraints ---
    vel_hulls = [
        time_to_spline_scale * h
        for h in minvo_hulls(control_points, derivative_order=1)
    ]
    acc_hulls = [
        (time_to_spline_scale**2) * h
        for h in minvo_hulls(control_points, derivative_order=2)
    ]

    constraints, lbg, ubg = build_constraints(
        symbolic_type,
        curve,
        start_conf,
        n_joints,
        vel_hulls,
        acc_hulls,
        start_limit=0.0,
        vel_limit=wmax,
        acc_limit=amax,
    )

    # --- Costs ---
    final_pos = robot.get_goal_point(curve[-1, :])

    cost_goal = get_cost_goal(
        final_pos,
        goal_position,
        weight=weights["goal"],
    )

    cost_jerk = get_cost_jerk(
        dddcurve,
        weight=weights["jerk"],
    )

    convolution_functor = ConvolutionFunctorWarp(
        "conv_drone",
        3,
        num_samples * 3,
        obstacle_means,
        covs_det,
        covs_inv,
    )

    cost_obstacles = weights["obstacle"] * convolution_functor(collision_points)

    total_cost = cost_goal + cost_obstacles + cost_jerk

    nlp = {
        "x": decision_variables,
        "f": total_cost,
        "p": params,
        "g": constraints,
    }

    solver = cas.nlpsol("drone_solver", "ipopt", nlp, _get_ipopt_options())
    return solver, lbg, ubg, convolution_functor
