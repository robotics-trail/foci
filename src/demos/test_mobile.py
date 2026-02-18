import numpy as np
import casadi as cas

from src.planning.planner import Planner
from src.visualization.visualizer import Visualizer


def mobile_robot_demo():
    urdf_path = "urdfs/ur5_extended_move.urdf"

    obstacle_means = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [2.0, 2.5, 1.0],
        ]
    )

    obstacle_covs = np.array(
        [
            np.diag([0.12**2, 0.12**2, 0.6**2]),
            np.diag([0.12**2, 0.12**2, 0.6**2]),
            np.diag([0.12**2, 0.12**2, 0.6**2]),
        ]
    )

    # robot_cov = np.eye(3) * 0.2**2
    robot_cov = np.array(
        [
            np.eye(3) * 0.0**2,
            np.eye(3) * 0.0**2,
            np.eye(3) * 0.0**2,
            np.eye(3) * 0.1**2,
            np.eye(3) * 0.1**2,
            np.eye(3) * 0.25**2,
            np.eye(3) * 0.25**2,
            np.eye(3) * 0.1**2,
            np.eye(3) * 0.1**2,
            np.eye(3) * 0.1**2,
            np.eye(3) * 0.1**2,
        ]
    )

    w_jerk = 0.00001
    w_goal = 100.0
    w_obstacle = 0.01  # w_jerk * alfa + w_goal * beta + w_obstacle * c
    # alfa + w_goal' * beta + w_obstacle' * c

    planner = Planner(
        urdf_file=urdf_path,
        root_link="world",
        tip_link="ee_link",
        obstacle_positions=obstacle_means,
        obstacle_covs=obstacle_covs,
        robot_cov=robot_cov,
        num_control_points=12,
        num_samples=25,
        weights={
            "jerk": w_jerk,
            "goal": w_goal,
            "obstacle": w_obstacle,
        },  # TODO: CAMBIAR A QUE DEPENDE DE 1 PESO
        wmax=5.0,
        vmax=5.0,
        amax=5.0,
    )

    theta_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ee_goal = np.array([3.0, 3.0, 4.5])

    curve = planner.plan(theta_start, ee_goal)

    vis = Visualizer(planner.robot, robot_cov, curve)
    vis.visualize_goal(ee_goal, radius=0.05)
    # vis.visualize_obstacles(table_means, table_covs, name="Table")
    vis.visualize_obstacles(obstacle_means, obstacle_covs, color=(100, 255, 100))
    vis.visualize_robot_gaussians()
    vis.visualize_trajectory()


if __name__ == "__main__":
    mobile_robot_demo()
