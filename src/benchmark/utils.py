import numpy as np


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
    min_dist = np.inf
    argmin = None

    for t, q in enumerate(trajectory):
        robot_points = np.asarray(robot.collision_points(q)).reshape(-1, 3)

        for i, pr in enumerate(robot_points):
            for j, po in enumerate(environment.obstacle_means):
                d = np.linalg.norm(pr - po)

                if d < min_dist:
                    min_dist = d
                    argmin = {
                        "sample": t,
                        "robot_gaussian": i,
                        "obstacle": j,
                        "robot_point": pr,
                        "obstacle_point": po,
                    }

    return min_dist, argmin