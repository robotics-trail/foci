import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
from src.robots.manipulator import ManipulatorRobot, LinkGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance, path_length
from src.benchmark.stomp_planner import STOMPPlanner
from src.benchmark.chomp_planner import CHOMPPlanner



def benchmark_bonsai(): 
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Bonsai.ply")
    
    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)

    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    translation = np.array([0.0, 2.5, 1.0])
    scale_factor = 3.0
    
    obstacle_means = (obstacle_means * scale_factor) @ rotation.T + translation
    obstacle_covs = np.einsum("ij,njk,lk->nil", rotation, obstacle_covs, rotation) * scale_factor**2
    
    obstacle_means = np.ascontiguousarray(obstacle_means, dtype=np.float32)
    obstacle_covs = np.ascontiguousarray(obstacle_covs, dtype=np.float32)


    gaussian_specs = [
        LinkGaussian("shoulder_link",   0.5, np.eye(3) * 0.1**2),
        LinkGaussian("upper_arm_link",  0.3, np.eye(3) * 0.2**2),
        LinkGaussian("upper_arm_link",  0.7, np.eye(3) * 0.2**2),  
        LinkGaussian("forearm_link",    0.5, np.eye(3) * 0.2**2),
        LinkGaussian("wrist_1_link",    0.5, np.eye(3) * 0.2**2),
        LinkGaussian("wrist_2_link",    0.5, np.eye(3) * 0.1**2),
    ]

    #theta_start = np.array([1.05, -0.23, -1.6, 1.21, -0.85, 0.02])
    theta_start = np.array([0.0, -np.pi/2.0, 0.0, -np.pi/2.0, np.pi/2.0, 0.0])
    
    goal = np.array([-1.5, 2.25, 0.5])

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
        real_wmax=4.0, 
        real_amax=3.0,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.02,
        goal_threshold=0.5,
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
            "goal": 2.0,
            "obstacle": 10.0,
            "jerk":0.1,
            "virtual_jerk": 0.01,
        },
        vmax=3.5,
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
    
    foci_min_dist, info = minimum_robot_environment_distance(robot,environment,foci_result.trajectory)
    chomp_min_dist, info = minimum_robot_environment_distance(robot,environment,chomp_result.trajectory)
    
    foci_path_length = path_length(foci_result.trajectory)
    chomp_path_length = path_length(chomp_result.trajectory)
    
    print("\n--- Planning timings FOCI---")
    print(f"Build:       {foci_result.timings['build']:.6f} s")
    print(f"Solver:      {foci_result.timings['solve']:.6f} s")
    print(f"Total:       {foci_result.timings['total']:.6f} s")
    print(f"Success:     {foci_result.success}")



    print("\n--- Planning timings CHOMP---")
    print(f"Build:       {chomp_result.timings['build']:.6f} s")
    print(f"Solver:      {chomp_result.timings['solve']:.6f} s")
    print(f"Total:       {chomp_result.timings['total']:.6f} s")
    print(f"Success:     {chomp_result.success}")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance FOCI: {foci_min_dist:.3f} m")
    print(f"Minimum robot-environment distance CHOMP: {chomp_min_dist:.3f} m")
    print(f"Path length FOCI: {foci_path_length:.3f} m")
    print(f"Path length CHOMP: {chomp_path_length:.3f} m")



    vis = RobotVisualizer(
        robot=robot,
        trajectory=foci_result.trajectory
    )

    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat("Bonsai", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path(name="FOCI")
    vis.visualize_initializer_path(foci_result.initial_trajectory, name="RRT*", color=(255, 0, 0))

    vis.visualize_initializer_path(chomp_result.trajectory, name="CHOMP", color=(0, 0, 255))
    vis.visualize_trajectory(loop=True)

            
if __name__ == "__main__":
    benchmark_bonsai()