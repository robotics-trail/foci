import numpy as np
import casadi as cas


def add_start_constraint(curve, start_values, cons, lbg, ubg, ltol=0.0, utol=1e-4):
    start_constraint = (
        (curve[0, 0] - start_values[0]) ** 2
        + (curve[0, 1] - start_values[1]) ** 2
        + (curve[0, 2] - start_values[2]) ** 2
    )

    lbg = np.concatenate((lbg, [ltol]))
    ubg = np.concatenate((ubg, [utol]))

    cons = cas.vertcat(cons, start_constraint)

    return cons, lbg, ubg


def add_hulls_constraints(hulls, cons, lbg, ubg, ltol=0.0, utol=1e-4):
    for hull in hulls:
        for i in range(hull.shape[0]):
            cons = cas.vertcat(
                cons, hull[i, 0] ** 2 + hull[i, 1] ** 2 + hull[i, 2] ** 2
            )
            lbg = np.concatenate((lbg, [ltol]))
            ubg = np.concatenate((ubg, [utol]))

    return cons, lbg, ubg


def build_constraints(
    SYM_TYPE,
    curve,
    start_values,
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
        curve, start_values, cons, lbg, ubg, ltol=0.0, utol=start_limit
    )

    # cons, lbg, ubg = add_hulls_constraints(
    #     pos_hulls, cons, lbg, ubg, ltol=0.0, utol=pos_limit
    # )
    cons, lbg, ubg = add_hulls_constraints(
        vel_hulls, cons, lbg, ubg, ltol=0.0, utol=vel_limit
    )
    cons, lbg, ubg = add_hulls_constraints(
        acc_hulls, cons, lbg, ubg, ltol=0.0, utol=acc_limit
    )

    return cons, lbg, ubg
