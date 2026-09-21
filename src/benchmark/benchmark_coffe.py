import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data
from src.robots.manipulator import ManipulatorRobot, LinkGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance
from src.benchmark.stomp_planner import STOMPPlanner
from src.benchmark.chomp_planner import CHOMPPlanner


def benchmark_coffee():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Coffee_Table.ply")

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data(ply_file)

    translation = np.array([1.2, 0.0, 0.6])
    scale_factor = 0.01
    mesh_scale = 1.3

    floor_height_threshold = np.min(obstacle_means[:, 2]) + 0.1
    mask = obstacle_means[:, 2] > floor_height_threshold

    obstacle_means = obstacle_means[mask]
    obstacle_covs = obstacle_covs[mask]
    colors = colors[mask]
    opacities = opacities[mask]

    centroid = obstacle_means.mean(axis=0)
    obstacle_means = (obstacle_means - centroid) * mesh_scale + centroid + translation
    obstacle_covs = obstacle_covs * (scale_factor * mesh_scale) ** 2

    obstacle_means = np.ascontiguousarray(obstacle_means, dtype=np.float32)
    obstacle_covs = np.ascontiguousarray(obstacle_covs, dtype=np.float32)

    gaussian_specs = [
        LinkGaussian("base_link",          0.5, np.eye(3) * 0.01**2),
        LinkGaussian("base_link_inertia",  0.5, np.eye(3) * 0.01**2),
        LinkGaussian("shoulder_link",      0.3, np.eye(3) * 0.04**2),
        LinkGaussian("shoulder_link",      0.7, np.eye(3) * 0.04**2),
        LinkGaussian("upper_arm_link",     0.3, np.eye(3) * 0.03**2),
        LinkGaussian("upper_arm_link",     0.7, np.eye(3) * 0.03**2),
        LinkGaussian("forearm_link",       0.5, np.eye(3) * 0.03**2),
        LinkGaussian("wrist_1_link",       0.5, np.eye(3) * 0.01**2),
        LinkGaussian("wrist_2_link",       0.5, np.eye(3) * 0.01**2),
        LinkGaussian("wrist_3_link",       0.5, np.eye(3) * 0.01**2),
    ]

    theta_start = np.array([-0.3, -1.2, 1.8, -2.1, -1.57, 0.0])
    goal = np.array([0.55, 0.0, 0.75])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5/ur5.urdf",
        root_link="base_link",
        tip_link="wrist_3_link",
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
        virtual_indices=[],
        virtual_wmax=4.0,
        virtual_amax=3.5,
        real_wmax=3.5,
        real_amax=3.0,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.0001,
        goal_threshold=0.001,
        random_seed=42,
        max_time=None,
    )

    foci_planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=12,
        num_samples=25,
        weights={
            "goal": 10.0,
            "obstacle": 1.0,
            "jerk": 0.001,
            "virtual_jerk": 0.01,
        },
        vmax=2.0,
        linear_solver="ma27",
    )

    chomp_planner = CHOMPPlanner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        num_waypoints=25,
        max_iter=1_000,
        learning_rate=0.01,
        weights={"obstacle": 1.0, "smoothness": 1.0},
        convergence_tol=1e-3,
    )

    stomp_planner = STOMPPlanner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        num_waypoints=12,
        n_samples=25,
        max_iter=800,
        temperature=5.19788696,
        weights={
            "obstacle": 127.98492000,
            "jerk": 0.00476016,
            "constraint": 4.05415357,
        },
        noise_scale=0.01096718,
        convergence_tol=0.005,
        total_time=8.0,
    )

    foci_result = foci_planner.plan(
        start=theta_start,
        goal=goal,
    )

    init_trajectory = np.ascontiguousarray(foci_result.initial_trajectory, dtype=np.float32)

    theta_final = np.array(foci_result.trajectory[-1], dtype=np.float32, copy=True)

    chomp_result = chomp_planner.plan(
        start=theta_start,
        goal=theta_final,
        initial_trajectory=init_trajectory,
    )

    stomp_result = stomp_planner.plan(
        start=np.ascontiguousarray(theta_start, dtype=np.float32),
        goal=theta_final,
        initial_trajectory=init_trajectory.copy(),
    )

    foci_min_dist, info = minimum_robot_environment_distance(robot, environment, foci_result.trajectory)
    stomp_min_dist, info = minimum_robot_environment_distance(robot, environment, stomp_result.trajectory)
    chomp_min_dist, info = minimum_robot_environment_distance(robot, environment, chomp_result.trajectory)

    print("\n--- Planning timings FOCI---")
    print(f"Build:       {foci_result.timings['build']:.6f} s")
    print(f"Solver:      {foci_result.timings['solve']:.6f} s")
    print(f"Total:       {foci_result.timings['total']:.6f} s")
    print(f"Success:     {foci_result.success}")

    print("\n--- Planning timings STOMP---")
    print(f"Build:       {stomp_result.timings['build']:.6f} s")
    print(f"Solver:      {stomp_result.timings['solve']:.6f} s")
    print(f"Total:       {stomp_result.timings['total']:.6f} s")
    print(f"Success:     {stomp_result.success}")

    print("\n--- Planning timings CHOMP---")
    print(f"Build:       {chomp_result.timings['build']:.6f} s")
    print(f"Solver:      {chomp_result.timings['solve']:.6f} s")
    print(f"Total:       {chomp_result.timings['total']:.6f} s")
    print(f"Success:     {chomp_result.success}")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance FOCI: {foci_min_dist:.3f} m")
    print(f"Minimum robot-environment distance STOMP: {stomp_min_dist:.3f} m")
    print(f"Minimum robot-environment distance CHOMP: {chomp_min_dist:.3f} m")

    vis = RobotVisualizer(
        robot=robot,
        trajectory=foci_result.trajectory,
    )

    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat("Coffee Table", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path(name="FOCI")
    vis.visualize_initializer_path(stomp_result.trajectory, name="STOMP", color=(255, 0, 255))
    vis.visualize_initializer_path(chomp_result.trajectory, name="CHOMP", color=(0, 0, 255))
    vis.visualize_trajectory(loop=True)


if __name__ == "__main__":
    benchmark_coffee()