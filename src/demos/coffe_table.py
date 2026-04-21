import os

import numpy as np

from src.utils.ply import extract_splat_data
from src.planning.planner import Planner, MultipleGaussiansPlanner
from src.planning.config import ProblemConfig, PlannerWeights, PlannerLimits
from src.planning.initializer import RRTStarConfig
from src.visualization.visualizer import Visualizer, MultipleGaussiansVisualizer


def coffe_table_demo(
    ply_file: str,
    urdf_path: str,
    ee_goal: np.ndarray,
    scale_factor: float = 0.05,
    subsample_rate: float = 0.1,
    translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
):

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data(ply_file)

    altura_min = obstacle_means[:, 2].min()
    altura_max = obstacle_means[:, 2].max()
    altura_total = altura_max - altura_min

    obstacle_means = obstacle_means * scale_factor + translation
    obstacle_covs = obstacle_covs * scale_factor**2

    robot_cov = np.eye(3) * 0.1**2

    gaussians_per_link = [
        (3, [0.5]),
    ]

    theta_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

    planner = MultipleGaussiansPlanner(
        config=problem_config, initializer_config=initializer_config
    )

    curve = planner.plan(theta_start, ee_goal)

    vis = MultipleGaussiansVisualizer(
        planner.robot,
        robot_cov,
        curve,
        gaussians_per_link,
        # ignore_link_indices=ignore_link_indices,
    )
    vis.visualize_goal(ee_goal, radius=0.05)
    vis.visualize_gaussian_splat(
        name="Coffe table",
        means=obstacle_means,
        covariances=obstacle_covs,
        colors=colors,
        opacities=opacities,
    )
    vis.visualize_trajectory()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Coffee_Table.ply")

    urdf_path = "urdfs/ur5.urdf"

    coffe_table_demo(
        ply_file,
        urdf_path,
        ee_goal=np.array([0.5, 0.5, 0.8]),
        scale_factor=1.5,
        subsample_rate=0.1,
        translation=np.array([0.0, 1.5, 1.0]),
    )

    # Para visualizar en navegador: http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/videos/prueba.viser
