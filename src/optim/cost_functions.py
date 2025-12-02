import casadi as cas


def get_cost_goal(position, target_position, weight=1.0):
    return weight * cas.sum1((position - target_position) ** 2)


def get_cost_obstacles(midpoints, convolution_functor, weight=1.0):
    return weight * convolution_functor(midpoints)


def get_cost_jerk(dddcurve, weight=1.0):
    return weight * cas.sum1(cas.sum2(dddcurve**2))
