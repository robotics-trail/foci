import numpy as np
import casadi as cas

from src.planning.initializer import RRTStarConfig
from src.planning.config import ProblemConfig, PlannerWeights, PlannerLimits
from src.planning.planner import Planner, MultipleGaussiansPlanner, RRTStarPlanner
from src.visualization.visualizer import MultipleGaussiansVisualizer


def mobile_robot_demo():
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

    colors = np.array([[100, 255, 100], [100, 255, 100], [100, 255, 100]])
    opacities = np.array([[0.5], [0.5], [0.5]])

    # robot_cov = np.eye(3) * 0.2**2
    robot_cov = np.array(
        [
            np.eye(3) * 0.1**2,
            np.eye(3) * 0.1**2,
            np.eye(3) * 0.1**2,
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

    gaussians_per_link = [
        (4, [0.3, 0.5, 0.7]),
        (5, [0.2, 0.5, 0.8]),
        (6, [0.5]),
        (7, [0.5]),
        (8, [0.5, 0.7]),
    ]

    ignore_link_indices = [0, 1, 2]

    theta_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ee_goal = np.array([3.0, 3.0, 4.5])

    planner_weights = PlannerWeights(jerk=0.00001, goal=100.0, obstacle=0.01)
    planner_limits = PlannerLimits(wmax=2.0, vmax=5.0, amax=2.0)

    initializer_config = RRTStarConfig(solve_time=3.0, random_seed=42)
    problem_config = ProblemConfig(
        urdf_file="urdfs/ur5_extended_move.urdf",
        root_link="world",
        tip_link="ee_link",
        obstacle_positions=obstacle_means,
        obstacle_covs=obstacle_covs,
        robot_cov=robot_cov,
        theta_start=theta_start,
        ee_goal=ee_goal,
        num_control_points=12,
        num_samples=25,
        weights=planner_weights,
        limits=planner_limits,
        ignore_link_indices=ignore_link_indices,
        gaussians_per_link=gaussians_per_link,
    )

    planner = MultipleGaussiansPlanner(
        config=problem_config, initializer_config=initializer_config
    )
    # planner = RRTStarPlanner(
    #     config=problem_config, initializer_config=initializer_config
    # )

    curve, timings = planner.plan()

    print("\n--- Planning timings ---")
    print(f"RRT initializer: {timings['initializer_rrt_time']:.6f} s")
    print(f"Solver:          {timings['solver_time']:.6f} s")
    print(f"Total:           {timings['total_time']:.6f} s")

    # print("\n--- Planning timings ---")
    # print(f"Solver:          {timings['rrt_time']:.6f} s")
    # print(f"Total:           {timings['total_time']:.6f} s")

    vis = MultipleGaussiansVisualizer(
        planner.robot,
        robot_cov,
        curve,
        gaussians_per_link,
        ignore_link_indices=ignore_link_indices,
    )
    vis.visualize_goal(ee_goal, radius=0.05)
    # vis.visualize_obstacles(table_means, table_covs, name="Table")
    vis.visualize_obstacles(obstacle_means, obstacle_covs, color=(100, 255, 100))
    # vis.visualize_gaussian_splat(
    #     "Test", obstacle_means, obstacle_covs, colors, opacities
    # )
    vis.visualize_robot_gaussians()
    vis.visualize_trajectory()


if __name__ == "__main__":
    mobile_robot_demo()
