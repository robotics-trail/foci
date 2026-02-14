import numpy as np
import casadi as cas

from src.planning.planner import Planner
from src.visualization.visualizer import Visualizer


def test_visualizer():
    urdf_path = "urdfs/ur5_extended.urdf"
    obstacle_means = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.35, -0.45, -0.575],
            [0.35, -0.45, -0.575],
            [-0.35, 0.45, -0.575],
            [0.35, 0.45, -0.575],
            [1.0, 0.1, 1.15],
            # [1.0, 0.25, 1.0],
        ]
    )
    obstacle_covs = np.array(
        [
            np.diag([0.7**2, 0.4**2, 0.05**2]),
            np.diag([0.05**2, 0.05**2, 0.37**2]),
            np.diag([0.05**2, 0.05**2, 0.37**2]),
            np.diag([0.05**2, 0.05**2, 0.37**2]),
            np.diag([0.05**2, 0.05**2, 0.37**2]),
            np.diag([0.12**2, 0.12**2, 0.12**2]),
            # np.diag([0.12**2, 0.12**2, 0.12**2]),
        ]
    )

    # obstacle_means = np.array([[1.0, 0.1, 1.15], [1.0, 0.25, 1.0]])
    # obstacle_covs = np.array(
    #     [np.diag([0.12**2, 0.12**2, 0.12**2]), np.diag([0.12**2, 0.12**2, 0.12**2])]
    # )

    robot_cov = np.eye(3) * 0.2**2

    planner = Planner(
        urdf_file=urdf_path,
        root_link="base_link",
        tip_link="ee_link",
        obstacle_positions=obstacle_means,
        obstacle_covs=obstacle_covs,
        robot_cov=robot_cov,
        num_control_points=12,
        num_samples=25,
        weights={
            "jerk": 0.00001,
            "goal": 100.0,
            "obstacle": 0.01,
        },  # TODO: CAMBIAR A QUE DEPENDE DE 1 PESO
        wmax=5.0,
        vmax=5.0,
        amax=5.0,
    )

    theta_start = np.array([0.0, -1.2, 1.2, -1.5, 0.0, 0.0])
    ee_goal = np.array([1.35, -0.25, 1.35])

    curve = planner.plan(theta_start, ee_goal)

    vis = Visualizer(planner.robot, robot_cov, curve)
    vis.visualize_goal(ee_goal, radius=0.05)
    vis.visualize_obstacles(obstacle_means, obstacle_covs, color=(100, 255, 100))
    # vis.visualize_trajectory(save_recording=False)
    vis.visualize_trajectory(
        save_recording=True, recording_path="videos/trajectory.viser"
    )


if __name__ == "__main__":
    test_visualizer()
