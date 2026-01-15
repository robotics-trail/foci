import os

import numpy as np

from src.utils.ply import extract_splat_data
from src.planning.planner import Planner
from src.visualization.visualizer import Visualizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ply_file = os.path.join(PROJECT_ROOT, "data", "Coffee_Table.ply")


# ============================== Extract data from ply file ==============================
# read in ply file
urdf_path = "urdfs/ur5_extended.urdf"

obstacle_means, obstacle_covs, colors, opacities = extract_splat_data(ply_file)

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

scale_factor = 0.05  # ajusta según la escena

subset = np.random.choice(len(obstacle_means), size=int(len(obstacle_means)*0.1), replace=False)
means = obstacle_means[subset]
covs = obstacle_covs[subset]
colors = colors[subset]
opacities = opacities[subset]

covs = covs * scale_factor**2

vis = Visualizer(planner.robot, robot_cov, curve)
vis.visualize_goal(ee_goal, radius=0.05)
# vis.visualize_obstacles(table_means, table_covs, name="Table")
#vis.visualize_obstacles(obstacle_means, obstacle_covs, color=(100, 255, 100))
vis.add_gaussians(means, covs, color = colors)
vis.visualize_trajectory()