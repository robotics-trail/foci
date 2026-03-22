import numpy as np
import casadi as cas


def _append_scalar_constraint(
    constraints,
    lower_bounds,
    upper_bounds,
    expr,
    lower_bound: float,
    upper_bound: float,
):
    """
    Append one scalar constraint and its bounds to the NLP containers.
    """
    constraints = cas.vertcat(constraints, expr)
    lower_bounds = np.concatenate((lower_bounds, [lower_bound]))
    upper_bounds = np.concatenate((upper_bounds, [upper_bound]))
    return constraints, lower_bounds, upper_bounds


def add_start_constraint(
    curve,
    start_values,
    n_joints: int,
    constraints,
    lower_bounds,
    upper_bounds,
    lower_bound: float = 0.0,
    upper_bound: float = 1e-4,
):
    """
    Constrain the first spline sample to stay close to the desired start configuration.

    The constraint is formulated as the squared Euclidean distance between the
    first spline point and the provided start configuration.

    Parameters
    ----------
    curve : cas.MX
        Spline samples of shape (num_samples, n_joints).
    start_values : cas.MX or np.ndarray
        Desired start configuration of shape (n_joints,).
    n_joints : int
        Number of robot joints.
    constraints : cas.MX
        Current stacked constraint vector.
    lower_bounds : np.ndarray
        Lower bounds associated with `constraints`.
    upper_bounds : np.ndarray
        Upper bounds associated with `constraints`.
    lower_bound : float, default=0.0
        Lower bound for the scalar start constraint.
    upper_bound : float, default=1e-4
        Upper bound for the scalar start constraint.

    Returns
    -------
    tuple
        Updated `(constraints, lower_bounds, upper_bounds)`.
    """
    start_constraint = sum(
        (curve[0, joint_idx] - start_values[joint_idx]) ** 2
        for joint_idx in range(n_joints)
    )

    return _append_scalar_constraint(
        constraints,
        lower_bounds,
        upper_bounds,
        start_constraint,
        lower_bound,
        upper_bound,
    )


def add_hulls_constraints(
    hulls,
    n_joints: int,
    constraints,
    lower_bounds,
    upper_bounds,
    lower_bound: float = 0.0,
    upper_bound: float = 1e-4,
):
    """
    Constrain every point of each hull using its squared joint-space norm.

    Each row of each hull is interpreted as one joint-space vector, and the
    scalar constraint added is `sum_j hull[i, j]^2`.

    Parameters
    ----------
    hulls : list
        List of hull matrices. Each hull has shape `(n_points, n_joints)`.
    n_joints : int
        Number of robot joints.
    constraints : cas.MX
        Current stacked constraint vector.
    lower_bounds : np.ndarray
        Lower bounds associated with `constraints`.
    upper_bounds : np.ndarray
        Upper bounds associated with `constraints`.
    lower_bound : float, default=0.0
        Lower bound for each scalar hull constraint.
    upper_bound : float, default=1e-4
        Upper bound for each scalar hull constraint.

    Returns
    -------
    tuple
        Updated `(constraints, lower_bounds, upper_bounds)`.
    """
    for hull in hulls:
        for row_idx in range(hull.shape[0]):
            hull_constraint = sum(
                hull[row_idx, joint_idx] ** 2 for joint_idx in range(n_joints)
            )

            constraints, lower_bounds, upper_bounds = _append_scalar_constraint(
                constraints,
                lower_bounds,
                upper_bounds,
                hull_constraint,
                lower_bound,
                upper_bound,
            )

    return constraints, lower_bounds, upper_bounds


def build_constraints(
    symbolic_type,
    curve,
    start_values,
    n_joints: int,
    vel_hulls,
    acc_hulls,
    start_limit: float,
    vel_limit: float,
    acc_limit: float,
):
    """
    Build the full nonlinear constraint vector and its bounds.

    The resulting constraints include:
    - start-configuration matching
    - velocity hull norm bounds
    - acceleration hull norm bounds

    Parameters
    ----------
    symbolic_type : type
        CasADi symbolic type, typically `cas.MX`.
    curve : cas.MX
        Spline samples of shape `(num_samples, n_joints)`.
    start_values : cas.MX or np.ndarray
        Desired start joint configuration.
    n_joints : int
        Number of robot joints.
    vel_hulls : list
        MINVO hulls for the first spline derivative.
    acc_hulls : list
        MINVO hulls for the second spline derivative.
    start_limit : float
        Upper bound for the squared start matching error.
    vel_limit : float
        Upper bound for the squared norm of each velocity hull point.
    acc_limit : float
        Upper bound for the squared norm of each acceleration hull point.

    Returns
    -------
    tuple
        `(constraints, lower_bounds, upper_bounds)`.
    """
    constraints = symbolic_type([])
    lower_bounds = np.array([], dtype=float)
    upper_bounds = np.array([], dtype=float)

    constraints, lower_bounds, upper_bounds = add_start_constraint(
        curve,
        start_values,
        n_joints,
        constraints,
        lower_bounds,
        upper_bounds,
        lower_bound=0.0,
        upper_bound=start_limit,
    )

    constraints, lower_bounds, upper_bounds = add_hulls_constraints(
        vel_hulls,
        n_joints,
        constraints,
        lower_bounds,
        upper_bounds,
        lower_bound=0.0,
        upper_bound=vel_limit,
    )

    constraints, lower_bounds, upper_bounds = add_hulls_constraints(
        acc_hulls,
        n_joints,
        constraints,
        lower_bounds,
        upper_bounds,
        lower_bound=0.0,
        upper_bound=acc_limit,
    )

    return constraints, lower_bounds, upper_bounds
