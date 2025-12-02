import numpy as np
import casadi as cas


def add_start_constraint(
    curve, start_values, n_joints, cons, lbg, ubg, ltol=0.0, utol=1e-4
):
    start_constraint = sum(
        (curve[0, j] - start_values[j]) ** 2 for j in range(n_joints)
    )

    lbg = np.concatenate((lbg, [ltol]))
    ubg = np.concatenate((ubg, [utol]))

    cons = cas.vertcat(cons, start_constraint)

    return cons, lbg, ubg


def add_hulls_constraints(hulls, n_joints, cons, lbg, ubg, ltol=0.0, utol=1e-4):
    for hull in hulls:
        for i in range(hull.shape[0]):
            hull_constraint = sum((hull[i, j]) ** 2 for j in range(n_joints))

            lbg = np.concatenate((lbg, [ltol]))
            ubg = np.concatenate((ubg, [utol]))

            cons = cas.vertcat(cons, hull_constraint)

    return cons, lbg, ubg


def build_constraints(
    SYM_TYPE,
    curve,
    start_values,
    n_joints,
    # pos_hulls,
    vel_hulls,
    acc_hulls,
    start_limit,
    # pos_limit,
    vel_limit,
    acc_limit,
):

    cons = SYM_TYPE([])
    lbg, ubg = [], []

    cons, lbg, ubg = add_start_constraint(
        curve, start_values, n_joints, cons, lbg, ubg, ltol=0.0, utol=start_limit
    )

    # cons, lbg, ubg = add_hulls_constraints(
    #     pos_hulls, cons, lbg, ubg, ltol=0.0, utol=pos_limit
    # )
    cons, lbg, ubg = add_hulls_constraints(
        vel_hulls, n_joints, cons, lbg, ubg, ltol=0.0, utol=vel_limit
    )
    cons, lbg, ubg = add_hulls_constraints(
        acc_hulls, n_joints, cons, lbg, ubg, ltol=0.0, utol=acc_limit
    )

    return cons, lbg, ubg
