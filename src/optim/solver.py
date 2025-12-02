import numpy as np
import casadi as cas

from src.core.robot_loader import ManipulatorRobotURDF
from src.core.convolution import ConvolutionFunctorWarp

from src.splines.bspline import BSpline
from src.splines.minvo import minvo_hulls

from src.optim.constraints import build_constraints
from src.optim.cost_functions import get_cost_goal, get_cost_obstacles, get_cost_jerk


def create_solver(
    robot: ManipulatorRobotURDF,
    num_control_points,
    obstacle_means,
    covs_det,
    covs_inv,
    num_samples=30,
    weights={"jerk": 0.1, "goal": 40.0, "obstacle": 40.0},
    wmax=1.0,
    vmax=1.0,
    amax=1.0,
):

    SYM_TYPE = cas.MX
    n_joints = robot.get_n_joints()
    n_links = robot.get_n_links()
    n_midpoints = n_links

    # --- Decision variables ---
    control_points = SYM_TYPE.sym("control_points", num_control_points, n_joints)
    dec_vars = cas.vertcat(cas.vec(control_points))

    # --- Parameteres ---
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

    dcurve = m_t_to_s * bspline.spline_eval(num_samples, der=1)
    ddcurve = (m_t_to_s**2) * bspline.spline_eval(num_samples, der=2)
    dddcurve = (m_t_to_s**3) * bspline.spline_eval(num_samples, der=3)

    # --- Kinematics ---
    q_sym = cas.MX.sym("q", n_joints)
    fk_function = cas.Function("fk", [q_sym], [robot.forward_kinematics(q_sym)])

    kinematics_functor = fk_function.map(num_samples, "openmp")
    joint_positions = kinematics_functor(curve.T).T
    joint_positions_reshaped = cas.reshape(
        joint_positions, num_samples * (n_links + 1), 3
    )

    midpoints = SYM_TYPE.zeros(num_samples * n_midpoints, 3)
    for sample_idx in range(num_samples):
        for segment_idx in range(n_midpoints):
            joint1_idx = sample_idx * (n_links + 1) + segment_idx
            joint2_idx = sample_idx * (n_links + 1) + segment_idx + 1

            midpoint_idx = sample_idx * n_midpoints + segment_idx
            midpoints[midpoint_idx, :] = (
                joint_positions_reshaped[joint1_idx, :]
                + joint_positions_reshaped[joint2_idx, :]
            ) / 2.0

    # --- Hulls ---
    vel_hulls = [m_t_to_s * h for h in minvo_hulls(control_points, derivative=1)]
    acc_hulls = [(m_t_to_s**2) * h for h in minvo_hulls(control_points, derivative=2)]

    # --- Constraints ---
    cons, lbg, ubg = build_constraints(
        SYM_TYPE,
        curve,
        start_conf,
        vel_hulls,
        acc_hulls,
        start_limit=0.05,
        vel_limit=wmax**2,
        acc_limit=amax**2,
    )

    # --- Costs ---
    final_ee_pos = robot.get_ee_endpoint(curve[-1, :])
    convolution_functor = ConvolutionFunctorWarp(
        "conv",
        3,
        n_links * num_samples,
        obstacle_means,
        covs_det,
        covs_inv,
    )

    cost_goal = get_cost_goal(final_ee_pos, goal_ee_position, weight=weights["goal"])
    cost_obstacles = get_cost_obstacles(
        midpoints, convolution_functor, weight=weights["obstacle"]
    )
    cost_jerk = get_cost_jerk(dddcurve, weight=weights["jerk"])

    total_cost = cost_goal + cost_obstacles + cost_jerk

    # --- NLP ---
    nlp = {"x": dec_vars, "f": total_cost, "p": params, "g": cons}
    ipopt_options = {
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

    solver = cas.nlpsol("solver", "ipopt", nlp, ipopt_options)

    return solver, lbg, ubg, convolution_functor
