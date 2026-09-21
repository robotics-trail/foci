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
from src.benchmark.utils import minimum_robot_environment_distance
from src.benchmark.stomp_planner import STOMPPlanner
from src.benchmark.chomp_planner import CHOMPPlanner

def random_search_stomp(
    robot,
    environment,
    joint_groups,
    start,
    goal,
    initial_trajectory,
    n_trials=15,
    random_seed=42,
    stop_on_success=True,
):
    """
    Small random search over the main STOMP hyperparameters.

    Configurations are ranked according to:
        1. STOMP convergence/success.
        2. Minimum robot-environment distance.
        3. Solver time.

    Returns
    -------
    best_result:
        Best STOMP planning result.

    best_params:
        Hyperparameters used by the best result.

    history:
        Information from every executed trial.
    """
    rng = np.random.default_rng(random_seed)

    start = np.ascontiguousarray(start, dtype=np.float32)
    goal = np.ascontiguousarray(goal, dtype=np.float32)
    initial_trajectory = np.ascontiguousarray(
        initial_trajectory,
        dtype=np.float32,
    )

    def log_uniform(low, high):
        """Sample uniformly in logarithmic space."""
        return float(
            np.exp(
                rng.uniform(
                    np.log(low),
                    np.log(high),
                )
            )
        )

    best_result = None
    best_params = None
    best_score = None
    history = []

    print("\n" + "=" * 80)
    print("STOMP RANDOM SEARCH")
    print("=" * 80)

    for trial_idx in range(n_trials):
        # Seed NumPy as well, in case STOMP uses np.random internally.
        np.random.seed(random_seed + trial_idx)

        params = {
            "n_samples": int(rng.choice([25, 50, 75, 100])),
            "max_iter": int(rng.choice([200, 400, 800])),
            "temperature": log_uniform(0.5, 20.0),
            "obstacle_weight": log_uniform(10.0, 1000.0),
            "jerk_weight": log_uniform(1e-4, 1e-1),
            "constraint_weight": log_uniform(1.0, 100.0),
            "noise_scale": log_uniform(0.01, 0.20),
            "convergence_tol": float(
                rng.choice([1e-2, 5e-3, 1e-3, 5e-4])
            ),
            "total_time": float(rng.choice([6.0, 8.0, 10.0])),
        }

        print(f"\n--- Trial {trial_idx + 1}/{n_trials} ---")
        print(
            f"n_samples={params['n_samples']}, "
            f"max_iter={params['max_iter']}"
        )
        print(
            f"temperature={params['temperature']:.5f}, "
            f"noise_scale={params['noise_scale']:.5f}"
        )
        print(
            f"weights: obstacle={params['obstacle_weight']:.5f}, "
            f"jerk={params['jerk_weight']:.6f}, "
            f"constraint={params['constraint_weight']:.5f}"
        )
        print(
            f"convergence_tol={params['convergence_tol']}, "
            f"total_time={params['total_time']}"
        )

        try:
            stomp_planner = STOMPPlanner(
                robot=robot,
                environment=environment,
                joint_groups=joint_groups,
                num_waypoints=initial_trajectory.shape[0],
                n_samples=params["n_samples"],
                max_iter=params["max_iter"],
                temperature=params["temperature"],
                weights={
                    "obstacle": params["obstacle_weight"],
                    "jerk": params["jerk_weight"],
                    "constraint": params["constraint_weight"],
                },
                noise_scale=params["noise_scale"],
                convergence_tol=params["convergence_tol"],
                total_time=params["total_time"],
            )

            result = stomp_planner.plan(
                start=start,
                goal=goal,
                initial_trajectory=initial_trajectory.copy(),
            )

            trajectory = np.asarray(result.trajectory, dtype=float)

            trajectory_is_valid = (
                trajectory.ndim == 2
                and trajectory.shape[1] == robot.n_dof
                and np.all(np.isfinite(trajectory))
            )

            if trajectory_is_valid:
                min_distance, _ = minimum_robot_environment_distance(
                    robot,
                    environment,
                    trajectory,
                )
                min_distance = float(min_distance)
            else:
                min_distance = -np.inf

            converged = bool(result.success) and trajectory_is_valid

            solve_time = float(
                result.timings.get("solve", np.inf)
            )
            total_runtime = float(
                result.timings.get("total", np.inf)
            )

            # Tuple comparison is lexicographical:
            # 1. Prefer converged trials.
            # 2. Prefer larger minimum distance.
            # 3. Prefer lower solve time.
            score = (
                int(converged),
                min_distance,
                -solve_time,
            )

            trial_data = {
                "trial": trial_idx + 1,
                "success": converged,
                "min_distance": min_distance,
                "solve_time": solve_time,
                "total_time": total_runtime,
                "params": params.copy(),
                "error": None,
            }

            history.append(trial_data)

            print(f"Success:      {converged}")
            print(f"Min distance: {min_distance:.6f} m")
            print(f"Solve time:   {solve_time:.6f} s")
            print(f"Total time:   {total_runtime:.6f} s")

            if best_score is None or score > best_score:
                best_score = score
                best_result = result
                best_params = params.copy()

                print("New best STOMP configuration.")

            if converged and stop_on_success:
                print("\nA converged STOMP configuration was found.")
                break

        except Exception as error:
            history.append(
                {
                    "trial": trial_idx + 1,
                    "success": False,
                    "min_distance": -np.inf,
                    "solve_time": np.inf,
                    "total_time": np.inf,
                    "params": params.copy(),
                    "error": repr(error),
                }
            )

            print(f"Trial failed with error: {error}")

    if best_result is None:
        raise RuntimeError(
            "All STOMP random-search trials failed."
        )

    print("\n" + "=" * 80)
    print("BEST STOMP CONFIGURATION")
    print("=" * 80)

    for param_name, param_value in best_params.items():
        if isinstance(param_value, float):
            print(f"{param_name}: {param_value:.8f}")
        else:
            print(f"{param_name}: {param_value}")

    print(f"Success: {best_result.success}")
    print(
        "Solve time: "
        f"{best_result.timings.get('solve', np.nan):.6f} s"
    )
    print(
        "Total time: "
        f"{best_result.timings.get('total', np.nan):.6f} s"
    )

    return best_result, best_params, history



def benchmark_bonsai(): 
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ply_file = os.path.join(PROJECT_ROOT, "data", "Bonsai.ply")
    
    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(ply_file)

    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    translation = np.array([0.0, 1.8, 1.0])
    scale_factor = 2.5
    
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
        LinkGaussian("wrist_3_link",    0.5, np.eye(3) * 0.1**2),
    ]

    theta_start = np.array([1.05, -0.23, -1.6, 1.21, -0.85, 0.02])
    #theta_start = np.array([0.0, -np.pi/2.0, 0.0, -np.pi/2.0, np.pi/2.0, 0.0])
    
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
    foci_min_dist, info = minimum_robot_environment_distance(robot,environment,foci_result.trajectory)
    stomp_min_dist, info = minimum_robot_environment_distance(robot,environment,stomp_result.trajectory)
    chomp_min_dist, info = minimum_robot_environment_distance(robot,environment,chomp_result.trajectory)
    
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
        trajectory=foci_result.trajectory
    )

    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat("Bonsai", obstacle_means, obstacle_covs, colors, opacities)
    vis.visualize_robot_gaussians()
    vis.visualize_path(name="FOCI")
    vis.visualize_initializer_path(stomp_result.trajectory, name="STOMP", color=(255, 0, 255))
    vis.visualize_initializer_path(chomp_result.trajectory, name="CHOMP", color=(0, 0, 255))
    vis.visualize_trajectory(loop=True)

            
if __name__ == "__main__":
    benchmark_bonsai()