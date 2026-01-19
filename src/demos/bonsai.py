import os

import numpy as np

from src.utils.ply import extract_splat_data_2
from src.planning.planner import Planner
from src.visualization.visualizer import Visualizer


def bonsai_demo(ply_file: str, urdf_path: str, ee_goal: np.ndarray, scale_factor: float = 0.05, subsample_rate: float = 0.1, translation: np.ndarray = np.array([0.0, 0.0, 0.0])):
    
    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)
    
    obstacle_means += translation
    obstacle_covs = obstacle_covs# * scale_factor**2
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

    theta_start = np.zeros(planner.n_joints)
    curve = planner.plan(theta_start, ee_goal)

   # subset = np.random.choice(len(obstacle_means), size=int(len(obstacle_means)*subsample_rate), replace=False)
   # means = obstacle_means[subset]
   # covs = obstacle_covs[subset]
   # colors = colors[subset]
   # opacities = opacities[subset]


    vis = Visualizer(planner.robot, robot_cov, curve)
    vis.visualize_goal(ee_goal, radius=0.05)
    vis.add_gaussians(obstacle_means, obstacle_covs, color = colors)
    vis.visualize_trajectory()

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Bonsai.ply")

    urdf_path = "urdfs/ur5_extended.urdf"

    bonsai_demo(ply_file, urdf_path, ee_goal = np.array([1.35, 1.25, 1.0]), scale_factor=10.0, subsample_rate=1.0, translation=np.array([1.0, 1.0, 0.3]))