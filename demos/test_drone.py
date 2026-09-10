import numpy as np

from src.robots.drone import DroneRobot, DroneGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance
from src.benchmark.stomp_planner import STOMPPlanner


def drone_demo():
    obstacle_means = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [2.0, 2.5, 1.0],
        ]
    )

    obstacle_covs = np.array(
        [
            np.diag([0.12**2, 0.12**2, 0.6**2]),
            np.diag([0.12**2, 0.12**2, 0.6**2]),
            np.diag([0.12**2, 0.12**2, 0.6**2]),
        ]
    )


    gaussian_specs = [DroneGaussian(np.array([0.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.1 ** 2)]

    theta_start = np.array([0.0, 0.0, 0.0, 0.0])
    goal = np.array([4.0, 3.0, 2.5])

    robot = DroneRobot(
        urdf_path="urdfs/drone_example.urdf",
        arm_length=0.15,
        gaussian_specs=gaussian_specs,
        xyz_limits=[(-0.5, 4.5), (-0.5, 3.5),(0.0, 3.0)]
    )

    joint_groups = JointGroups(
        virtual_indices=[0,1,2,3], 
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
        voxel_size=0.1,
        goal_threshold=0.005,
        random_seed=10,
        max_time=None,   
)
    rrt_result = initializer.initialize(robot, environment, theta_start, goal, num_control_points=40)
    
    
    planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=12,
        num_samples=25,
        weights={
            "goal": 10.0,
            "obstacle": 0.01,
            "jerk": 0.00049327,
            "virtual_jerk": 0.00049327,
        },
        vmax=5.0,
        linear_solver="mumps"
    )

    stomp = STOMPPlanner(
        robot=robot, 
        environment=environment,
        joint_groups=joint_groups, 
        num_waypoints=40, 
        n_samples=25,
        max_iter=800,
        temperature=10.0,
        convergence_tol=1e-3,
        noise_scale= 0.1,
        seed=42,
        total_time= 5.0,
    )


    result = planner.plan(
        start=theta_start,
        goal=goal,
    )

    theta_final = result.trajectory[-1]

    # stomp_result = stomp.plan(
    #     start=theta_start,
    #     goal=theta_final,
    #     initial_trajectory=result.initial_trajectory,
    # )

    min_dist_foci, info = minimum_robot_environment_distance(robot,environment,result.trajectory)
    # min_dist_stomp, info = minimum_robot_environment_distance(robot,environment,stomp_result.trajectory)

    print("\n--- Planning timings FOCI---")
    print(f"Initializer: {result.timings['initializer']:.6f} s")
    print(f"Build:       {result.timings['build']:.6f} s")
    print(f"Solver:      {result.timings['solve']:.6f} s")
    print(f"Total:       {result.timings['total']:.6f} s")
    print(f"Success:     {result.success}")

    print("\n--- Planning timings STOMP---")
    # print(f"Initializer: {stomp_result.timings['initializer']:.6f} s")
    # print(f"Build:       {stomp_result.timings['build']:.6f} s")
    # print(f"Solver:      {stomp_result.timings['solve']:.6f} s")
    # print(f"Total:       {stomp_result.timings['total']:.6f} s")
    # print(f"Success:     {stomp_result.success}")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance FOCI: {min_dist_foci:.3f} m")
    # print(f"Minimum robot-environment distance CHOMP: {min_dist_stomp:.3f} m")

    vis = RobotVisualizer(
    robot=robot,
    trajectory=result.trajectory,

)

    vis.visualize_goal(goal)
    vis.visualize_obstacles(obstacle_means, obstacle_covs)
    vis.visualize_robot_gaussians()
    vis.visualize_path()
    # vis.visualize_initializer_path(result.trajectory, name="FOCI")
    vis.visualize_initializer_path(result.initial_trajectory, color=(0, 0, 255), name="RRT*")
    vis.visualize_trajectory(loop=True)

    

if __name__ == "__main__":
    drone_demo()
