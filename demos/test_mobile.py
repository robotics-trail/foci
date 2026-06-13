import numpy as np

from src.robots.mobile import MobileRobot, MobileGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer
from src.benchmark.utils import minimum_robot_environment_distance


def mobile_demo():
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


    gaussian_specs = [MobileGaussian(np.array([0.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.3 ** 2)]


    theta_start = np.array([0.0, 0.0, 0.0])
    goal = np.array([4.0, 3.0, 0.5])

    robot = MobileRobot(
        urdf_path="urdfs/anymal.urdf",
        body_center_height=1.0,
        gaussian_specs=gaussian_specs,
        xy_limits=[(-0.5, 4.5), (-0.5, 3.5)]
    )

    joint_groups = JointGroups(
        virtual_indices=[0,1], 
        virtual_wmax=5.0, 
        virtual_amax=2.5, 
        real_wmax=2.0, 
        real_amax=1.5,
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
    
    planner = Planner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=initializer,
        num_control_points=12,
        num_samples=25,
        weights={
            "goal": 5.0,
            "obstacle": 0.05,
            "jerk": 0.001,
            "virtual_jerk": 0.001,
        },
        vmax=5.0,
        linear_solver="mumps"
    )


    result = planner.plan(
        start=theta_start,
        goal=goal,
    )

    min_dist_foci, info = minimum_robot_environment_distance(robot,environment,result.trajectory)

    print("\n--- Planning timings FOCI---")
    print(f"Initializer: {result.timings['initializer']:.6f} s")
    print(f"Build:       {result.timings['build']:.6f} s")
    print(f"Solver:      {result.timings['solve']:.6f} s")
    print(f"Total:       {result.timings['total']:.6f} s")
    print(f"Success:     {result.success}")

    print("\n--- Benchmark metrics ---")
    print(f"Number of environment gaussians: {len(obstacle_means)}")
    print(f"Number of robot gaussians: {len(gaussian_specs)}")
    print(f"Minimum robot-environment distance FOCI: {min_dist_foci:.3f} m")

    vis = RobotVisualizer(
    robot=robot,
    trajectory=result.trajectory,
    )

    vis.visualize_goal(goal)
    vis.visualize_obstacles(obstacle_means, obstacle_covs)
    vis.visualize_robot_gaussians()
    vis.visualize_path()
    vis.visualize_initializer_path(result.initial_trajectory, color=(0, 0, 255), name="RRT*")
    vis.visualize_trajectory(loop=True)

    

if __name__ == "__main__":
    mobile_demo()
