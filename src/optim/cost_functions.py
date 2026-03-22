import casadi as cas


def get_cost_goal(position, target_position, weight: float = 1.0):
    """
    Quadratic end-effector goal cost.

    Parameters
    ----------
    position : cas.MX
        Current end-effector position of shape `(3, 1)` or compatible.
    target_position : cas.MX or np.ndarray
        Desired end-effector position.
    weight : float, default=1.0
        Scalar weight for the term.

    Returns
    -------
    cas.MX
        Weighted squared Euclidean distance to the goal.
    """
    return weight * cas.sum1((position - target_position) ** 2)


def get_cost_obstacles(
    collision_points,
    convolution_functor,
    multiple_gaussians: bool,
    n_active_links: int,
    weight: float = 1.0,
):
    """
    Obstacle avoidance cost based on Gaussian convolution.

    Parameters
    ----------
    collision_points : cas.MX
        Evaluation points used to measure obstacle proximity.
        In the simplest case these are link midpoints; in more detailed models
        they may be arbitrary Gaussian points along links.
    convolution_functor : callable or list[callable]
        Convolution functor(s) used to evaluate obstacle cost.
    multiple_gaussians : bool
        If False, a single covariance model is shared across links.
        If True, one convolution functor per active link is used.
    n_active_links : int
        Number of active links considered in the obstacle model.
    weight : float, default=1.0
        Scalar weight for the term.

    Returns
    -------
    cas.MX
        Weighted obstacle cost.
    """
    if not multiple_gaussians:
        return weight * convolution_functor(collision_points)

    convolution_sum = 0
    for link_idx, conv_func in enumerate(convolution_functor):
        link_points = collision_points[link_idx::n_active_links, :]
        convolution_sum += conv_func(link_points)

    return weight * convolution_sum / n_active_links


def get_cost_jerk(dddcurve, weight: float = 1.0):
    """
    Quadratic jerk regularization cost.

    Parameters
    ----------
    dddcurve : cas.MX
        Third derivative of the spline, shape `(num_samples, n_joints)`.
    weight : float, default=1.0
        Scalar weight for the term.

    Returns
    -------
    cas.MX
        Weighted sum of squared jerk values.
    """
    return weight * cas.sum1(cas.sum2(dddcurve**2))
