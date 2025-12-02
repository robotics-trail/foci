import numpy as np
import casadi as cas

from src.planning.planner import Planner
from src.visualization.visualizer import Visualizer


def test_visualizer():
    urdf_path = "urdfs/demo_arm.urdf"
    obstacle_means = np.array([[2.0, 2.0, 1.0], [5.0, 3.0, 2.0]])
    obstacle_covs = np.array([np.eye(3) * 0.5, np.diag([0.7, 0.4, 0.6])])

    robot_cov = np.eye(3) * 0.2

    planner = Planner(
        urdf_file=urdf_path,
        root_link="base_link",
        tip_link="link3",
        obstacle_positions=obstacle_means,
        obstacle_covs=obstacle_covs,
        robot_cov=robot_cov,
        num_control_points=8,
        num_samples=15,
        weights={"jerk": 0.00001, "goal": 100.0, "obstacle": 0.01},
        wmax=5.0,
        vmax=5.0,
        amax=5.0,
    )

    theta_start = np.zeros(planner.n_joints)
    ee_goal = np.array([2.0, 2.0, 5.0])

    curve = planner.plan(theta_start, ee_goal)

    vis = Visualizer(planner.robot, robot_cov, curve)
    vis.visualize_goal(ee_goal)
    # vis.visualize_obstacles(obstacle_means, obstacle_covs)
    vis.visualize_trajectory()


if __name__ == "__main__":
    test_visualizer()
