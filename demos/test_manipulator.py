import numpy as np

from src.robots.manipulator import ManipulatorRobot, LinkGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance
from src.benchmark.chomp_planner import CHOMPPlanner

def mobile_robot_demo():
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

    gaussian_specs = [ 
        LinkGaussian(4, 0.7, np.eye(3) * 0.1**2), 
        LinkGaussian(5, 0.2, np.eye(3) * 0.1**2), 
        LinkGaussian(5, 0.5, np.eye(3) * 0.1**2), 
        LinkGaussian(5, 0.8, np.eye(3) * 0.1**2), 
        LinkGaussian(6, 0.5, np.eye(3) * 0.2**2), 
        LinkGaussian(7, 0.5, np.eye(3) * 0.1**2), 
        LinkGaussian(8, 0.5, np.eye(3) * 0.1**2), 
    ]

    theta_start = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    goal = np.array([3.0, 3.0, 4.0])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5_extended_move.urdf",
        root_link="world",
        tip_link="ee_link",
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
        virtual_indices=[0,1,2], 
        virtual_wmax=4.5, 
        virtual_amax=5.0, 
        real_wmax=4.5, 
        real_amax=5.5,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )


    initializer = RRTStarInitializer(
        voxel_size=0.1,
        goal_threshold=0.001,
        random_seed=10,
        max_time=None,   
)
    

    planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=15,
        num_samples=30,
        weights={
            "goal": 10.0,
            "obstacle": 0.01,
            "jerk": 0.001,
            "virtual_jerk": 0.001,
        },
        vmax=4.5,
        linear_solver="mumps",
    )

    
    result = planner.plan(
        start=theta_start,
        goal=goal,
    )

    theta_final = result.trajectory[-1]

#     chomp = CHOMPPlanner(
#     robot=robot,
#     environment=environment,
#     joint_groups=joint_groups,
#     num_waypoints=40,
#     max_iter=300,
#     learning_rate=0.01,
#     weights={
#         "smoothness": 1.0,
#         "obstacle": 1.0,
#     },
# )

#     chomp_result = chomp.plan(
#         start=theta_start,
#         goal=theta_final,
#         initial_trajectory=result.initial_trajectory,
#     )

    min_dist_foci, info = minimum_robot_environment_distance(robot,environment,result.trajectory)
    # min_dist_chomp, info = minimum_robot_environment_distance(robot,environment,chomp_result.trajectory)

    print("\n--- Planning timings FOCI---")
    print(f"Initializer: {result.timings['initializer']:.6f} s")
    print(f"Build:       {result.timings['build']:.6f} s")
    print(f"Solver:      {result.timings['solve']:.6f} s")
    print(f"Total:       {result.timings['total']:.6f} s")
    print(f"Success:     {result.success}")

    # print("\n--- Planning timings CHOMP---")
    # print(f"Initializer: {chomp_result.timings['initializer']:.6f} s")
    # print(f"Build:       {chomp_result.timings['build']:.6f} s")
    # print(f"Solver:      {chomp_result.timings['solve']:.6f} s")
    # print(f"Total:       {chomp_result.timings['total']:.6f} s")
    # print(f"Success:     {chomp_result.success}")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance FOCI: {min_dist_foci:.3f} m")
    # print(f"Minimum robot-environment distance CHOMP: {min_dist_chomp:.3f} m")


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
    mobile_robot_demo()