import os
import sys

from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data_2
from src.robots.drone import DroneGaussian, DroneRobot
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import (
    jerk_metric,
    limit_scaling_factor,
    minimum_robot_environment_distance,
    path_length,
    resample_trajectory,
)
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

    _t = perf_counter()
    foci_planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
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

    foci_build_time = perf_counter() - _t

    _t = perf_counter()
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
        seed=42,
        total_time=2.0,
    )

    stomp_build_time = perf_counter() - _t

    _t = perf_counter()
    chomp_planner = CHOMPPlanner(
        robot=robot, 
        environment=environment, 
        joint_groups=joint_groups, 
        num_waypoints=12,
        max_iter=1_000, 
        learning_rate=0.01, 
        weights={"obstacle": 1.0, "smoothness": 1.0},
        convergence_tol=1e-3,
        total_time=2.0,
    )

    chomp_build_time = perf_counter() - _t

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


    # ---- Metrics ------------------------------------------------------
    # One function per quantity for the three planners, all evaluated at the
    # same resolution.  Every path and derivative metric is resolution
    # dependent, and the waypoint planners have nothing finer than their own
    # waypoints to offer, so num_waypoints is the common ground.
    metric_samples = 12

    # Same duration the NLP assumes: ||goal - f_task(start)|| / vmax.
    foci_duration = max(
        float(
            np.linalg.norm(goal - np.asarray(robot.f_task(theta_start)).ravel())
            / foci_planner.vmax
        ),
        1e-3,
    )

    runs = (
        ("FOCI",  foci_result,  foci_duration,  foci_build_time),
        ("STOMP", stomp_result, stomp_result.metadata["total_time"], stomp_build_time),
        ("CHOMP", chomp_result, chomp_result.metadata["total_time"], chomp_build_time),
    )

    print("\n--- Benchmark metrics ---")
    print(f"Environment gaussians: {len(obstacle_means)}"
          f"   robot gaussians: {len(gaussian_specs)}"
          f"   metric samples: {metric_samples}")
    print(
        f"{'planner':<7} {'wall [s]':>9} {'min dist [m]':>13} {'length':>9} "
        f"{'jerk':>11} {'T [s]':>7} {'T feasible':>11} {'x limits':>9} {'success':>8}"
    )
    print("-" * 92)

    for name, result, duration, build_time in runs:
        # Wall time includes construction: FOCI builds its NLP and instantiates
        # IPOPT inside plan(), while the baselines prepare their CasADi
        # callbacks and matrices in __init__, so their own `timings` are not
        # comparable with each other.
        wall = build_time + result.timings["total"]

        xi = resample_trajectory(result.trajectory, metric_samples)
        min_dist, _ = minimum_robot_environment_distance(robot, environment, xi)
        scale, feasible_duration = limit_scaling_factor(
            xi, duration, joint_groups, robot.n_dof, metric_samples
        )

        print(
            f"{name:<7} {wall:9.3f} {min_dist:13.3f} {path_length(xi):9.3f} "
            f"{jerk_metric(xi, duration, metric_samples):11.3e} {duration:7.2f} "
            f"{feasible_duration:11.2f} {scale:9.2f} {str(result.success):>8}"
        )

    print(
        "\n'T feasible' is the duration each trajectory would need to respect "
        "wmax/amax;\n'x limits' is the factor over the nominal duration "
        "(1.00 = already feasible)."
    )

    # Opt in with --vis: the visualizer loops forever and would
    # otherwise block the benchmark.
    visualize_stomp = "--vis" in sys.argv

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