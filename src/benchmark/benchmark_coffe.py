import sys

from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.ply import extract_splat_data
from src.utils.paths import data_path
from src.robots.manipulator import ManipulatorRobot, LinkGaussian
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



def coffe_table_benchmark():
    ply_file = data_path("Coffee_Table.ply")

    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data(ply_file)

    translation = np.array([0.71, 0.0, 0.6])
    scale_factor = 0.005
    mesh_scale = 1.3
    
    floor_height_threshold = np.min(obstacle_means[:, 2]) + 0.1
    mask = obstacle_means[:, 2] > floor_height_threshold

    obstacle_means = obstacle_means[mask]
    obstacle_covs = obstacle_covs[mask]
    colors = colors[mask]
    opacities = opacities[mask]
    
    centroid = obstacle_means.mean(axis=0)
    obstacle_means = (obstacle_means - centroid) * mesh_scale + centroid + translation
    obstacle_covs = obstacle_covs * (scale_factor * mesh_scale)**2

    gaussian_specs = [ 
        LinkGaussian(0, 0.5, np.eye(3) * 0.01**2), 
        LinkGaussian(1, 0.5, np.eye(3) * 0.01**2), 
        LinkGaussian(2, 0.3, np.eye(3) * 0.04**2), 
        LinkGaussian(2, 0.7, np.eye(3) * 0.04**2), 
        LinkGaussian(3, 0.3, np.eye(3) * 0.03**2), 
        LinkGaussian(3, 0.7, np.eye(3) * 0.03**2), 
        LinkGaussian(4, 0.5, np.eye(3) * 0.03**2), 
        LinkGaussian(5, 0.5, np.eye(3) * 0.01**2), 
        LinkGaussian(6, 0.5, np.eye(3) * 0.01**2), 
        LinkGaussian(7, 0.5, np.eye(3) * 0.01**2), 
    ]

    theta_start = np.array([-0.3, -1.2, 1.8, -2.1, -1.57, 0.0])
    goal = np.array([0.55, 0.0, 0.75])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5.urdf",
        root_link="base_link",
        tip_link="ee_link",
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
        virtual_indices=[], 
        virtual_wmax=4.0, 
        virtual_amax=3.5, 
        real_wmax=7.5, 
        real_amax=6.5,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )

    initializer = RRTStarInitializer(
        voxel_size=0.01,
        goal_threshold=0.01,
        random_seed=42,
        max_time=None,   
    )

    _t = perf_counter()
    foci_planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=15,
        num_samples=30,
        weights={
            "goal": 150.0,
            "obstacle": 550.0,
            "jerk": 0.00237317,
            "virtual_jerk": 0.00593292,
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
        num_waypoints=15, 
        n_samples=30, 
        max_iter=1_000,
        temperature=10, 
        # _obstacle_cost_trajectory now averages over waypoints too, so its
        # scale matches FOCI's.  15.0 = num_waypoints keeps the previous
        # effective obstacle/jerk/constraint balance under the new units.
        weights={"obstacle": 15.0, "jerk": 1.0, "constraint": 1.0},
        noise_scale=0.1,
        convergence_tol=1e-3, 
        seed=42,
        total_time=5.0,
    )

    stomp_build_time = perf_counter() - _t

    _t = perf_counter()
    chomp_planner = CHOMPPlanner(
        robot=robot, 
        environment=environment, 
        joint_groups=joint_groups, 
        num_waypoints=15,
        max_iter=1_000, 
        learning_rate=0.01, 
        weights={"obstacle": 1.0, "smoothness": 1.0},
        convergence_tol=1e-3,
        total_time=5.0,
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
    metric_samples = 15

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

    # Visualisation is opt-in: the visualizer animates in an endless loop, so
    # opening it at all makes the benchmark impossible to run unattended.  This
    # used to be a `if --vis: show STOMP else: show CHOMP`, i.e. BOTH branches
    # blocked and the flag only picked the trajectory.  Now no flag means no
    # visualizer, `--vis` shows STOMP and `--vis --chomp` shows CHOMP.
    if "--vis" not in sys.argv:
        return

    name, shown = (("CHOMP", chomp_result) if "--chomp" in sys.argv
                   else ("STOMP", stomp_result))

    vis = RobotVisualizer(robot=robot, trajectory=shown.trajectory)
    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat("Coffee Table", obstacle_means, obstacle_covs,
                                 colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path(name=name)
    vis.visualize_initializer_path(shown.initial_trajectory)
    vis.visualize_initializer_path(foci_result.trajectory, name="FOCI",
                                   color=(0, 0, 255))
    vis.visualize_trajectory(loop=True)


if __name__ == "__main__":
    coffe_table_benchmark()
