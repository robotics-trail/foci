import numpy as np
import casadi as cas

from src.core.robot_loader import ManipulatorRobotURDF
from src.core.convolution import ConvolutionFunctorWarp

from src.splines.bspline import BSpline
from src.splines.minvo import minvo_hulls

from src.optim.constraints import build_constraints
from src.optim.cost_functions import get_cost_goal, get_cost_obstacles, get_cost_jerk


def _get_ipopt_options():
    return {
        "ipopt.print_level": 5,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-1,
        "print_time": 0,
        "ipopt.acceptable_tol": 1e-1,
        "ipopt.acceptable_obj_change_tol": 1e-1,
        "ipopt.constr_viol_tol": 1e-1,
        "ipopt.acceptable_iter": 1,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
    }


def _create_solver_common(
    robot,
    num_control_points,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians,
    active_link_indices,
    num_samples,
    weights,
    wmax,
    vmax,
    amax,
    point_builder,
    obstacle_cost_builder,
    solver_name,
    fk_name="fk",
):
    SYM_TYPE = cas.MX
    n_joints = robot.get_n_joints()
    n_links = robot.get_n_links()

    # --- Decision variables ---
    control_points = SYM_TYPE.sym("control_points", num_control_points, n_joints)
    dec_vars = cas.vertcat(cas.vec(control_points))

    # --- Parameters ---
    start_conf = SYM_TYPE.sym("start_conf", n_joints, 1)
    goal_ee_position = SYM_TYPE.sym("goal_ee_position", 3, 1)
    params = cas.vertcat(cas.vec(start_conf), cas.vec(goal_ee_position))

    # --- Spline evaluation ---
    bspline = BSpline(control_points)
    curve = bspline.spline_eval(num_samples)

    start_ee_position = robot.get_ee_endpoint(curve[0, :])
    T_val = cas.norm_2(goal_ee_position - start_ee_position) / vmax
    S = num_control_points - 4
    m_t_to_s = S / T_val

    dcurve = m_t_to_s * bspline.spline_eval(num_samples, derivative_order=1)
    ddcurve = (m_t_to_s**2) * bspline.spline_eval(num_samples, derivative_order=2)
    dddcurve = (m_t_to_s**3) * bspline.spline_eval(num_samples, derivative_order=3)

    # --- Kinematics ---
    q_sym = SYM_TYPE.sym("q", n_joints)
    fk_function = cas.Function(fk_name, [q_sym], [robot.forward_kinematics(q_sym)])
    kinematics_functor = fk_function.map(num_samples, "openmp")
    joint_positions = kinematics_functor(curve.T).T
    joint_positions_reshaped = cas.reshape(
        joint_positions, num_samples * (n_links + 1), 3
    )

    # --- Collision evaluation points ---
    collision_points, aux_data = point_builder(
        SYM_TYPE=SYM_TYPE,
        joint_positions_reshaped=joint_positions_reshaped,
        num_samples=num_samples,
        n_links=n_links,
        active_link_indices=active_link_indices,
    )

    # --- Hulls ---
    vel_hulls = [m_t_to_s * h for h in minvo_hulls(control_points, derivative_order=1)]
    acc_hulls = [
        (m_t_to_s**2) * h for h in minvo_hulls(control_points, derivative_order=2)
    ]

    # --- Constraints ---
    cons, lbg, ubg = build_constraints(
        SYM_TYPE,
        curve,
        start_conf,
        n_joints,
        vel_hulls,
        acc_hulls,
        start_limit=0.01,
        vel_limit=wmax**2,
        acc_limit=amax**2,
    )

    # --- Costs ---
    final_ee_pos = robot.get_ee_endpoint(curve[-1, :])
    cost_goal = get_cost_goal(final_ee_pos, goal_ee_position, weight=weights["goal"])
    cost_jerk = get_cost_jerk(dddcurve, weight=weights["jerk"])

    cost_obstacles, convolution_functor = obstacle_cost_builder(
        collision_points=collision_points,
        aux_data=aux_data,
        obstacle_means=obstacle_means,
        covs_det=covs_det,
        covs_inv=covs_inv,
        multiple_gaussians=multiple_gaussians,
        active_link_indices=active_link_indices,
        num_samples=num_samples,
        weights=weights,
    )

    total_cost = cost_goal + cost_obstacles + cost_jerk

    # --- NLP ---
    nlp = {"x": dec_vars, "f": total_cost, "p": params, "g": cons}
    solver = cas.nlpsol(solver_name, "ipopt", nlp, _get_ipopt_options())

    return solver, lbg, ubg, convolution_functor


def _build_midpoints(
    SYM_TYPE,
    joint_positions_reshaped,
    num_samples,
    n_links,
    active_link_indices,
):
    n_active_links = len(active_link_indices)
    midpoints = SYM_TYPE.zeros(num_samples * n_active_links, 3)

    for sample_idx in range(num_samples):
        base = sample_idx * (n_links + 1)

        for local_idx, link_idx in enumerate(active_link_indices):
            joint1_idx = base + link_idx
            joint2_idx = base + link_idx + 1

            midpoint_idx = sample_idx * n_active_links + local_idx
            midpoints[midpoint_idx, :] = (
                joint_positions_reshaped[joint1_idx, :]
                + joint_positions_reshaped[joint2_idx, :]
            ) / 2.0

    return midpoints, None


def _build_gaussian_points(
    SYM_TYPE,
    joint_positions_reshaped,
    num_samples,
    n_links,
    active_link_indices,
    gaussian_specs,
):
    n_total_gaussians = len(gaussian_specs)
    gaussian_points = SYM_TYPE.zeros(num_samples * n_total_gaussians, 3)
    points_by_link = {link_idx: [] for link_idx in active_link_indices}

    for sample_idx in range(num_samples):
        base_joint = sample_idx * (n_links + 1)

        for g_idx, (link_idx, t) in enumerate(gaussian_specs):
            p0 = joint_positions_reshaped[base_joint + link_idx, :]
            p1 = joint_positions_reshaped[base_joint + link_idx + 1, :]

            point = (1.0 - t) * p0 + t * p1

            flat_idx = sample_idx * n_total_gaussians + g_idx
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
    multiple_gaussians,
    active_link_indices,
    num_samples,
    weights,
):
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
        for k in range(n_active_links):
            conv_func = ConvolutionFunctorWarp(
                f"conv_link_{k}",
                3,
                num_samples,
                obstacle_means,
                covs_det[k],
                covs_inv[k],
            )
            convolution_functor.append(conv_func)

    cost_obstacles = get_cost_obstacles(
        collision_points,
        convolution_functor,
        multiple_gaussians,
        n_active_links,
        weight=weights["obstacle"],
    )

    return cost_obstacles, convolution_functor


def _build_obstacle_cost_multi_gauss(
    collision_points,
    aux_data,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians,
    active_link_indices,
    num_samples,
    weights,
):
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
        cost_obstacles = weights["obstacle"] * convolution_functor(collision_points)
        return cost_obstacles, convolution_functor

    active_link_to_local = {
        link_idx: k for k, link_idx in enumerate(active_link_indices)
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
        raise ValueError("No gaussian points assigned to active links.")

    cost_obstacles = weights["obstacle"] * (weighted_sum / total_points)
    return cost_obstacles, convolution_functor


def create_solver(
    robot: ManipulatorRobotURDF,
    num_control_points,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians,
    active_link_indices,
    num_samples=30,
    weights={"jerk": 1.0, "goal": 1.0, "obstacle": 1.0},
    wmax=1.0,
    vmax=1.0,
    amax=1.0,
):
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
    num_control_points,
    obstacle_means,
    covs_det,
    covs_inv,
    multiple_gaussians,
    active_link_indices,
    gaussian_specs,
    num_samples=30,
    weights={"jerk": 1.0, "goal": 1.0, "obstacle": 1.0},
    wmax=1.0,
    vmax=1.0,
    amax=1.0,
):
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
