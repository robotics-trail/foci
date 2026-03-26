import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
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

    theta_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    

    planner_weights = PlannerWeights(jerk=0.00001, goal=500.0, obstacle=0.05)
    planner_limits = PlannerLimits(wmax=5.0, vmax=5.0, amax=5.0)

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

    planner = MultipleGaussiansPlanner(config=problem_config)
    #planner = Planner(config=problem_config)
    curve = planner.plan()

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

    bonsai_demo(ply_file, urdf_path, ee_goal = np.array([0.2, 1.3, 2.0]), scale_factor=3.0, subsample_rate=1.0, translation=np.array([0.0, 1.5, 1.0]))