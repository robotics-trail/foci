import numpy as np

from src.robots.manipulator import ManipulatorRobot, LinkGaussian
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.planner import Planner
from src.planning.joints import JointGroups
from src.visualization.visualizer import RobotVisualizer

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
        LinkGaussian(4, 0.3, np.eye(3) * 0.2**2), 
        LinkGaussian(4, 0.5, np.eye(3) * 0.2**2), 
        LinkGaussian(4, 0.7, np.eye(3) * 0.2**2), 
        LinkGaussian(5, 0.2, np.eye(3) * 0.2**2), 
        LinkGaussian(5, 0.5, np.eye(3) * 0.2**2), 
        LinkGaussian(5, 0.8, np.eye(3) * 0.2**2), 
        LinkGaussian(6, 0.5, np.eye(3) * 0.1**2), 
        LinkGaussian(7, 0.5, np.eye(3) * 0.1**2), 
        LinkGaussian(8, 0.5, np.eye(3) * 0.1**2), 
    ]

    theta_start = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    goal = np.array([3.0, 3.0, 4.5])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5_extended_move.urdf",
        root_link="world",
        tip_link="ee_link",
        gaussian_specs=gaussian_specs,
    )

    joint_groups = JointGroups(
        virtual_indices=[0,1,2], 
        virtual_wmax=2.0, 
        virtual_amax=2.0, 
        real_wmax=2.0, 
        real_amax=2.0,
    )

    environment = GaussianEnvironment(
        obstacle_means=obstacle_means,
        obstacle_covariances=obstacle_covs,
    )


    initializer = RRTStarInitializer(
        voxel_size=0.1,
        goal_threshold=0.01,
        random_seed=42,
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
            "goal": 1.0,
            "obstacle": 1.0,
            "jerk": 1.0,
            "virtual_jerk": 1.0,
        },
        vmax=5.0,
    )

    result = planner.plan(
        start=theta_start,
        goal=goal,
    )

    print("\n--- Planning timings ---")
    print(f"Initializer: {result.timings['initializer']:.6f} s")
    print(f"Build:       {result.timings['build']:.6f} s")
    print(f"Solver:      {result.timings['solve']:.6f} s")
    print(f"Total:       {result.timings['total']:.6f} s")
    print(f"Success:     {result.success}")

    vis = RobotVisualizer(
    robot=robot,
    trajectory=result.trajectory,
)

    vis.visualize_goal(goal)
    vis.visualize_obstacles(obstacle_means, obstacle_covs)
    vis.visualize_robot_gaussians()
    vis.visualize_path()
    if result.initial_trajectory is not None:
        vis.visualize_initializer_path(result.initial_trajectory)

    vis.visualize_trajectory(loop=True)

    

if __name__ == "__main__":
    mobile_robot_demo()