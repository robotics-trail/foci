import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
from src.robots.drone import DroneGaussian, DroneRobot
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance
from src.benchmark.stomp_planner import STOMPPlanner
from src.benchmark.chomp_planner import CHOMPPlanner



def bonsai_demo(): 
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data/Forest.ply")

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)

    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    translation = np.array([8.0, 0.0, 0.0])
    scale_factor = 10.0
    
    
    obstacle_means = (obstacle_means * scale_factor) @ rotation.T + translation
    obstacle_covs = np.einsum("ij,njk,lk->nil", rotation, obstacle_covs, rotation) * scale_factor**2


    gaussian_specs = [DroneGaussian(np.array([0.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.1 ** 2)]

    theta_start = np.array([0.0, -3.0, 1.0, 0.0])
    goal = np.array([14.0, 5.0, 3.5])


    theta_start = np.array([1.05, -0.23, -1.6, 1.21, -0.85, 0.02])
    goal = np.array([-1.5, 2.25, 0.5])

    robot = DroneRobot(
        urdf_path="urdfs/drone_example.urdf",
        arm_length=0.15,
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
        virtual_indices=[0,1,2], 
        virtual_wmax=5.0, 
        virtual_amax=3.0, 
        real_wmax=5.0, 
        real_amax=3.0,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.01,
        goal_threshold=0.005,
        random_seed=42,
        max_time=None,   
    )

    foci_planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        #initializer=initializer,
        num_control_points=12,
        num_samples=25,
        weights={
            "goal": 100.0,
            "obstacle": 200.0,
            "jerk": 0.049327,
            "virtual_jerk": 0.00049327,
        },
        vmax=1.0,
        linear_solver="ma27"
    )

    stomp_planner = STOMPPlanner(
        robot=robot, 
        environment=environment, 
        joint_groups=joint_groups, 
        num_waypoints=12, 
        n_samples=25, 
        max_iter=1_000,
        temperature=10, 
        weights={"obstacle": 1.0, "jerk": 1.0, "constraint": 1.0},
        noise_scale=0.1,
        convergence_tol=1e-3, 
        total_time=2.0,
    )

    chomp_planner = CHOMPPlanner(
        robot=robot, 
        environment=environment, 
        joint_groups=joint_groups, 
        num_waypoints=12,
        max_iter=1_000, 
        learning_rate=0.01, 
        weights={"obstacle": 1.0, "smoothness": 1.0},
        convergence_tol=1e-3,            
    )

    foci_result = foci_planner.plan(
        start=theta_start,
        goal=goal,
    )

    theta_final = foci_result.trajectory[-1]

    stomp_result = stomp_planner.plan(
        start=theta_start,
        goal=theta_final,
        initial_trajectory=foci_result.initial_trajectory,
    )

    chomp_result = chomp_planner.plan(
        start=theta_start,
        goal=theta_final,
        initial_trajectory=foci_result.initial_trajectory,
    )


    stomp_min_dist, info = minimum_robot_environment_distance(robot,environment,stomp_result.trajectory)
    chomp_min_dist, info = minimum_robot_environment_distance(robot,environment,chomp_result.trajectory)

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
    print(f"Minimum robot-environment distance STOMP: {stomp_min_dist:.3f} m")
    print(f"Minimum robot-environment distance CHOMP: {chomp_min_dist:.3f} m")

    visualize_stomp = True

    if visualize_stomp: 
        vis = RobotVisualizer(
            robot=robot,
            trajectory=stomp_result.trajectory
        )

        vis.visualize_goal(goal)
        vis.visualize_gaussian_splat("Forest", obstacle_means, obstacle_covs, colors, opacities)
        vis.visualize_robot_gaussians()
        vis.visualize_path(name="STOMP")
        vis.visualize_initializer_path(stomp_result.initial_trajectory)
        vis.visualize_initializer_path(foci_result.trajectory, name="FOCI", color=(0, 0, 255))
        vis.visualize_trajectory(loop=True)

    else: 
        vis = RobotVisualizer(
            robot=robot,
            trajectory=chomp_result.trajectory
        )

        vis.visualize_goal(goal)
        vis.visualize_gaussian_splat("Forest", obstacle_means, obstacle_covs, colors, opacities)
        vis.visualize_robot_gaussians()
        vis.visualize_path(name="CHOMP")
        vis.visualize_initializer_path(chomp_result.initial_trajectory)
        vis.visualize_initializer_path(foci_result.trajectory, name="FOCI", color=(0, 0, 255))
        vis.visualize_trajectory(loop=True)