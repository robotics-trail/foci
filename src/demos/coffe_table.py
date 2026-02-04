import os

import numpy as np

from src.utils.ply import extract_splat_data
from src.planning.planner import Planner
from src.visualization.visualizer import Visualizer


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

    obstacle_means += translation
    obstacle_means *= scale_factor
    obstacle_covs = obstacle_covs * scale_factor**2

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

    print(
        f"Altura min: {altura_min}, Altura max: {altura_max}, Altura total: {altura_total}"
    )

    scene_center = (obstacle_means.mean(axis=0) + ee_goal) / 2
    camera_position = scene_center + np.array([2.0, 2.0, 1.5])

    vis = Visualizer(
        planner.robot,
        robot_cov,
        curve,
        camera_position=camera_position,
        camera_look_at=scene_center,
    )
    vis.visualize_goal(ee_goal, radius=0.05)
    vis.add_gaussians(obstacle_means, obstacle_covs, color=colors)
    vis.visualize_trajectory(
        save_recording=True, recording_path="videos/coffee_table.viser"
    )


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Coffee_Table.ply")

    urdf_path = "urdfs/ur5.urdf"

    coffe_table_demo(
        ply_file,
        urdf_path,
        ee_goal=np.array([0.5, 0.5, 0.8]),
        scale_factor=0.5,
        subsample_rate=0.1,
        translation=np.array([0.0, 0.0, 0.22]),
    )

    # Para visualizar en navegador: http://localhost:8000/viser-client/?playbackPath=http://localhost:8000/videos/prueba.viser
