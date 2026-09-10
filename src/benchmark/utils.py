"""Shared metrics for the planner benchmarks.

Everything a benchmark script compares across planners lives here, so the
three planners are measured with one function instead of each reporting its
own internal cost in its own units.

Resolution is part of every derivative metric here
--------------------------------------------------
`jerk_metric` and `limit_scaling_factor` both resample to `num_samples` and
then take finite differences.  A planner's output is a POLYLINE, so its second
and third derivatives are impulses at the corners and any finite-difference
estimate of them grows without bound as `num_samples` grows.  Measured on a
12-waypoint trajectory:

    num_samples   12     25     50    100    200
    max|acc|    17.1   20.0   43.9   93.0  170.0

Every planner in a comparison must therefore be measured at the SAME
`num_samples`, and that value must be the coarsest representation involved
(a waypoint planner has nothing finer to offer).  Pass the benchmark's
`num_waypoints` explicitly -- never rely on the default.
"""

import numpy as np


_EPS = 1e-12


def resample_trajectory(trajectory, num_samples: int) -> np.ndarray:
    """
    Resample a waypoint trajectory to `num_samples` points by linear
    interpolation in the normalised parameter [0, 1].

    Planners return trajectories at different resolutions (12 waypoints for
    STOMP/CHOMP, 25 spline samples for FOCI), and every path/derivative metric
    is resolution dependent, so metrics must be computed after resampling all
    of them to a common number of points.
    """
    trajectory = np.asarray(trajectory, dtype=float)

    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be a 2D array of shape (N, D), got {trajectory.shape}."
        )
    if trajectory.shape[0] < 2:
        raise ValueError(
            f"trajectory must contain at least two points, got {trajectory.shape[0]}."
        )
    if num_samples < 2:
        raise ValueError(f"num_samples must be at least 2, got {num_samples}.")
    if trajectory.shape[0] == num_samples:
        return trajectory.copy()

    old_s = np.linspace(0.0, 1.0, trajectory.shape[0])
    new_s = np.linspace(0.0, 1.0, num_samples)

    return np.vstack(
        [np.interp(new_s, old_s, trajectory[:, d]) for d in range(trajectory.shape[1])]
    ).T


_MIN_DERIVATIVE_SAMPLES: int = 3


def _central_derivative(xi: np.ndarray, dt: float, order: int) -> np.ndarray:
    """Central finite differences of a waypoint trajectory, edges replicated.

    Needs at least three rows: a central difference consumes one row at each
    end, and with two rows `inner` comes out empty, which collapses the result
    to shape (0, D) and every later `.max()` raises "zero-size array to
    reduction operation" from somewhere unrelated.  Fail here instead.
    """
    d = np.asarray(xi, dtype=float)

    if d.shape[0] < _MIN_DERIVATIVE_SAMPLES:
        raise ValueError(
            f"a central derivative needs at least {_MIN_DERIVATIVE_SAMPLES} "
            f"samples, got {d.shape[0]}."
        )

    for _ in range(order):
        inner = (d[2:] - d[:-2]) / (2.0 * max(dt, _EPS))
        d = np.vstack([inner[:1], inner, inner[-1:]])

    return d


def jerk_metric(trajectory, duration: float, num_samples: int = 200) -> float:
    """
    Integrated squared jerk of a trajectory, in physical units.

        J = integral ||d3q/dt3||^2 dt

    Computed after resampling to `num_samples` points, and the resolution is
    part of the metric: a coarse polyline underestimates the jerk of the
    smooth curve it samples (at 12 waypoints a 4-point stencil recovers only
    about 22% of the true value), so this is a comparison figure, not an
    absolute physical one.

    For that comparison to be fair, every planner must be measured at the SAME
    `num_samples`, which in practice means the coarsest representation
    involved -- a waypoint planner has nothing finer to offer, and upsampling
    its polyline would just put all the curvature in the corners.  Pass the
    benchmark's `num_waypoints`.

    This is deliberately NOT any planner's internal cost: those are in
    different units and are not comparable with each other.
    """
    if num_samples < _MIN_DERIVATIVE_SAMPLES:
        raise ValueError(
            f"num_samples must be at least {_MIN_DERIVATIVE_SAMPLES}, got "
            f"{num_samples}."
        )

    xi = resample_trajectory(trajectory, num_samples)
    dt = float(duration) / max(num_samples - 1, 1)
    jerk = _central_derivative(xi, dt, order=3)

    return float(dt * np.sum(jerk ** 2))


def limit_scaling_factor(
    trajectory,
    duration: float,
    joint_groups,
    n_dof: int,
    num_samples: int = 200,
) -> tuple[float, float]:
    """
    Smallest uniform time scaling that makes a trajectory respect the
    velocity and acceleration limits.

    Scaling the duration by s divides velocity by s and acceleration by s^2,
    and leaves the geometry untouched, so the factor is exact in one shot:

        s = max( max|v| / wmax , sqrt(max|a| / amax) , 1 )

    Planners that do not enforce the limits (CHOMP does not look at them at
    all, and STOMP only adds a soft penalty whose relative weight is
    arbitrary) can then be compared on equal footing: the returned duration is
    how long each one actually needs to be executable.

    `num_samples` is part of the metric, exactly as in `jerk_metric`: the
    acceleration of a polyline is an impulse at every corner, so max|acc| --
    and with it the returned scale -- grows with the resampling resolution.
    Always pass the benchmark's `num_waypoints`; leaving the default 200 on a
    12-waypoint trajectory inflates the scale by roughly 3x (see the module
    docstring).

    Returns
    -------
    (scale, feasible_duration)
        `scale` is 1.0 when the trajectory already satisfies the limits.
    """
    if num_samples < _MIN_DERIVATIVE_SAMPLES:
        raise ValueError(
            f"num_samples must be at least {_MIN_DERIVATIVE_SAMPLES}, got "
            f"{num_samples}."
        )

    xi = resample_trajectory(trajectory, num_samples)
    dt = float(duration) / max(num_samples - 1, 1)

    vel = _central_derivative(xi, dt, order=1)
    acc = _central_derivative(xi, dt, order=2)

    real_idx = list(joint_groups.real_indices(n_dof))
    virt_idx = list(joint_groups.virtual_indices)

    scales = [1.0]

    if real_idx:
        scales.append(np.abs(vel[:, real_idx]).max() / max(joint_groups.real_wmax, _EPS))
        scales.append(
            np.sqrt(np.abs(acc[:, real_idx]).max() / max(joint_groups.real_amax, _EPS))
        )

    if virt_idx:
        # Virtual joints are limited by the norm of their sub-vector, matching
        # the constraint FOCI imposes on the MINVO hulls.
        scales.append(
            np.linalg.norm(vel[:, virt_idx], axis=1).max()
            / max(joint_groups.virtual_wmax, _EPS)
        )
        scales.append(
            np.sqrt(
                np.linalg.norm(acc[:, virt_idx], axis=1).max()
                / max(joint_groups.virtual_amax, _EPS)
            )
        )

    scale = float(max(scales))

    return scale, scale * float(duration)


def path_length(trajectory) -> float:
    """
    Compute the length of a trajectory as the sum of the Euclidean
    distances between consecutive points.

    Works for any sequence of points (joint-space configurations,
    task-space positions, control points, ...), as long as it can be
    interpreted as an array of shape (N, D).

    Parameters
    ----------
    trajectory :
        Sequence of N points of dimension D (array-like of shape (N, D)).

    Returns
    -------
    float
        Total path length, i.e. sum_{i=0}^{N-2} || trajectory[i+1] - trajectory[i] ||.
        Returns 0.0 if the trajectory has fewer than 2 points.
    """
    trajectory = np.asarray(trajectory)

    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory must be a 2D array of shape (N, D), got shape {trajectory.shape}."
        )

    if trajectory.shape[0] < 2:
        return 0.0

    segment_vectors = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    return float(np.sum(segment_lengths))


def path_lengths(trajectories: dict) -> dict:
    """
    Compute the path length of several trajectories at once.

    Parameters
    ----------
    trajectories :
        Mapping from a name (e.g. planner name) to its trajectory.

    Returns
    -------
    dict
        Mapping from the same names to their path length.
    """
    return {name: path_length(trajectory) for name, trajectory in trajectories.items()}


def minimum_robot_environment_distance(robot, environment, trajectory):
    """
    Smallest centre-to-centre distance between any robot Gaussian and any
    obstacle Gaussian along the trajectory.

    Note this ignores the Gaussians' extents: it is a distance between
    centres, not a clearance to the surfaces.  It is the same quantity for
    every planner, so it compares fine.

    Vectorised over obstacles: the previous triple Python loop made the metric
    unusable on splat-sized environments (10^5 obstacles), which is why it was
    only ever evaluated at a dozen samples.
    """
    obstacle_means = np.asarray(environment.obstacle_means, dtype=float)

    min_dist = np.inf
    argmin = None

    for t, q in enumerate(trajectory):
        robot_points = np.asarray(robot.collision_points(q), dtype=float).reshape(-1, 3)

        # (n_robot_gaussians, n_obstacles)
        distances = np.linalg.norm(
            robot_points[:, None, :] - obstacle_means[None, :, :], axis=2
        )

        flat = int(np.argmin(distances))
        i, j = np.unravel_index(flat, distances.shape)
        d = float(distances[i, j])

        if d < min_dist:
            min_dist = d
            argmin = {
                "sample": t,
                "robot_gaussian": int(i),
                "obstacle": int(j),
                "robot_point": robot_points[i],
                "obstacle_point": obstacle_means[j],
            }

    return min_dist, argmin
