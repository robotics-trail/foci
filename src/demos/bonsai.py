import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
from src.planning.initializer import RRTStarConfig
from src.planning.config import ProblemConfig, PlannerWeights, PlannerLimits
from src.planning.planner import Planner, MultipleGaussiansPlanner
from src.visualization.visualizer import MultipleGaussiansVisualizer, Visualizer


def bonsai_demo(ply_file: str, urdf_path: str, ee_goal: np.ndarray, scale_factor: float = 0.05, subsample_rate: float = 0.1, translation: np.ndarray = np.array([0.0, 0.0, 0.0])):
    
    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)
    
    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    
    obstacle_means = (obstacle_means * scale_factor) @ rotation.T + translation
    obstacle_covs = np.einsum("ij,njk,lk->nil", rotation, obstacle_covs, rotation) * scale_factor**2

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
    
    robot_cov = np.eye(3) * 0.1**2

    gaussians_per_link = [
        (3, [0.5]),
    ]

    #ignore_link_indices = [0, 1, 2]

    #theta_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta_start = np.array([1.05, -0.23, -1.6, 1.21, -0.85, 0.02])
    

    planner_weights = PlannerWeights(jerk=0.00005, goal=800.0, obstacle=0.001)
    planner_limits = PlannerLimits(wmax=5.0, vmax=7.5, amax=4.5)

    
    initializer_config = RRTStarConfig(solve_time=5.0, random_seed=42)

    problem_config = ProblemConfig(
        urdf_file=urdf_path,
        root_link="base_link",
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

    planner = MultipleGaussiansPlanner(config=problem_config, initializer_config=initializer_config)    
    #planner = Planner(config=problem_config)
    curve, timings = planner.plan()
    
    print("\n--- Planning timings ---")
    print(f"RRT initializer: {timings['initializer_rrt_time']:.6f} s")
    print(f"Solver:          {timings['solver_time']:.6f} s")
    print(f"Total:           {timings['total_time']:.6f} s")


    print("Last configuration:", curve[-1])

    vis = MultipleGaussiansVisualizer(
        planner.robot,
        robot_cov,
        curve,
        gaussians_per_link,
        #ignore_link_indices=ignore_link_indices,
    )
    
    #vis = Visualizer(
        #planner.robot, 
        #robot_cov,
        #curve,
    #)
    
    vis.visualize_goal(ee_goal, radius=0.05)
    # vis.visualize_obstacles(table_means, table_covs, name="Table")
    #vis.visualize_obstacles(obstacle_means, obstacle_covs, color=(100, 255, 100))
    vis.server.add_gaussian_splats(name="Bonsai", centers=obstacle_means, covariances=obstacle_covs, rgbs=colors, opacities=opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_trajectory()

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Bonsai.ply")

    urdf_path = "urdfs/ur5.urdf"
    urdf_path = "urdfs/ur5_extended.urdf"

    bonsai_demo(ply_file, urdf_path, ee_goal = np.array([-2.0, 2.2, 0.5]), scale_factor=3.0, subsample_rate=1.0, translation=np.array([0.0, 1.5, 1.0]))