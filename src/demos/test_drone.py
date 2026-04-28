import numpy as np

from src.planning.config import PlannerLimits, PlannerWeights, ProblemConfig
from src.planning.initializer import RRTStarConfig
from src.planning.planner import DronePlanner

from src.visualization.visualizer import DroneVisualizer


def drone_demo():

    # -- Obstacle parameteres --
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

    # -- Robot parameteres --
    robot_cov = np.eye(3) * 0.2**2

    gaussians_per_link = [
        (1, [0.5]),
        (2, [0.5]),
        (3, [0.5]),
        (4, [0.5]),
        (5, [0.5]),
    ]

    # -- Planning parameters --
    theta_start = np.array([0.0, 0.0, 0.0, 0.0])
    ee_goal = np.array([3.0, 4.0, 4.0])

    planner_weights = PlannerWeights(jerk=0.00001, goal=300.0, obstacle=0.01)
    planner_limits = PlannerLimits(wmax=2.0, vmax=5.0, amax=2.0)

    problem_config = ProblemConfig(
        urdf_file="urdfs/drone_example.urdf",
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
        gaussians_per_link=gaussians_per_link,
    )

    planner = DronePlanner(problem_config)
    curve = planner.plan(theta_start, ee_goal)

    vis = DroneVisualizer(curve, "urdfs/drone_example.urdf")
    vis.visualize_goal(ee_goal, radius=0.08)
    vis.visualize_obstacles(obstacle_means, obstacle_covs, color=(255, 100, 100))
    vis.visualize_path(color=np.array([50, 200, 50]).reshape((3,)))
    vis.visualize_trajectory(dt=0.1, loop=True)


if __name__ == "__main__":
    drone_demo()
