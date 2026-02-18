import casadi as cas


def get_cost_goal(position, target_position, weight=1.0):
    return weight * cas.sum1((position - target_position) ** 2)


def get_cost_obstacles(
    midpoints, convolution_functor, multiple_gaussians, n_links, weight=1.0
):

    if not multiple_gaussians:
        return weight * convolution_functor(midpoints)

    else:
        sum_conv = 0
        for link_idx, conv_func in enumerate(convolution_functor):
            midpoints_link = midpoints[link_idx::n_links, :]
            sum_conv += conv_func(midpoints_link)

        return weight * sum_conv / n_links


def get_cost_jerk(dddcurve, weight=1.0):
    return weight * cas.sum1(cas.sum2(dddcurve**2))
