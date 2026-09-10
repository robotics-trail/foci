"""FOCI vs CHOMP vs STOMP on the Coffee Table splat scene.

    python demos/compare_coffe_table.py                    # metrics table only
    python demos/compare_coffe_table.py --vis              # + viser on http://localhost:8081
    python demos/compare_coffe_table.py --vis --port 9000  # pick the port
    python demos/compare_coffe_table.py --stomp-iters 1000 # STOMP's own default

STOMP is gradient-free: every iteration scores 25-30 noisy trajectories
against every environment Gaussian, which costs ~2.5 s/iteration here.  Its
default is capped below the benchmark's 1000 so the demo finishes; the table
prints the iteration count and whether it converged on its own.
"""

import argparse
import sys

import numpy as np

_DEFAULT_PORT = 8081

from src.benchmark.comparison import (
    print_metrics,
    run_comparison,
    save_runs,
    visualize_comparison,
)
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.joints import JointGroups
from src.robots.manipulator import LinkGaussian, ManipulatorRobot
from src.utils.paths import data_path
from src.utils.ply import extract_splat_data


def compare_coffe_table(stomp_iters: int = 200, chomp_iters: int = 1000):
    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data(
        data_path("Coffee_Table.ply")
    )

    translation = np.array([0.71, 0.0, 0.6])
    scale_factor = 0.005
    mesh_scale = 1.3

    # Drop the floor plane so the table itself is the obstacle.
    floor_height_threshold = np.min(obstacle_means[:, 2]) + 0.1
    mask = obstacle_means[:, 2] > floor_height_threshold

    obstacle_means = obstacle_means[mask]
    obstacle_covs = obstacle_covs[mask]
    colors = colors[mask]
    opacities = opacities[mask]

    centroid = obstacle_means.mean(axis=0)
    obstacle_means = (obstacle_means - centroid) * mesh_scale + centroid + translation
    obstacle_covs = obstacle_covs * (scale_factor * mesh_scale) ** 2

    gaussian_specs = [
        LinkGaussian(0, 0.5, np.eye(3) * 0.01 ** 2),
        LinkGaussian(1, 0.5, np.eye(3) * 0.01 ** 2),
        LinkGaussian(2, 0.3, np.eye(3) * 0.04 ** 2),
        LinkGaussian(2, 0.7, np.eye(3) * 0.04 ** 2),
        LinkGaussian(3, 0.3, np.eye(3) * 0.03 ** 2),
        LinkGaussian(3, 0.7, np.eye(3) * 0.03 ** 2),
        LinkGaussian(4, 0.5, np.eye(3) * 0.03 ** 2),
        LinkGaussian(5, 0.5, np.eye(3) * 0.01 ** 2),
        LinkGaussian(6, 0.5, np.eye(3) * 0.01 ** 2),
        LinkGaussian(7, 0.5, np.eye(3) * 0.01 ** 2),
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

    runs = run_comparison(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        start=theta_start,
        goal=goal,
        foci_options=dict(
            initializer=RRTStarInitializer(
                voxel_size=0.01, goal_threshold=0.01, random_seed=42,
            ),
            num_control_points=15,
            num_samples=30,
            weights={"goal": 150.0, "obstacle": 550.0,
                     "jerk": 0.00237317, "virtual_jerk": 0.00593292},
            vmax=1.0,
            linear_solver="ma27",
        ),
        chomp_options=dict(
            num_waypoints=15, max_iter=chomp_iters, learning_rate=0.01,
            weights={"obstacle": 1.0, "smoothness": 1.0},
            convergence_tol=1e-3, total_time=5.0,
        ),
        stomp_options=dict(
            num_waypoints=15, n_samples=30, max_iter=stomp_iters, temperature=10.0,
            # obstacle = num_waypoints, see compare_bonsai.py.
            weights={"obstacle": 15.0, "jerk": 1.0, "constraint": 1.0},
            noise_scale=0.1, convergence_tol=1e-3, seed=42, total_time=5.0,
        ),
    )

    save_runs("coffe_table_runs.npz", runs)
    print_metrics("Coffee Table", runs, environment, len(gaussian_specs))

    if "--vis" in sys.argv:
        port = (int(sys.argv[sys.argv.index("--port") + 1])
                if "--port" in sys.argv else _DEFAULT_PORT)
        print(f"\nviser: http://localhost:{port}   "
              "(FOCI green, CHOMP blue, STOMP red)", flush=True)
        visualize_comparison(
            port=port,
            runs=runs, robot=robot, goal=goal, splat_name="Coffee Table",
            obstacle_means=obstacle_means, obstacle_covs=obstacle_covs,
            colors=colors, opacities=opacities,
        )


def _cli():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vis", action="store_true",
                        help="serve the viser visualisation after the table")
    parser.add_argument("--port", type=int, default=8081,
                        help="viser port (default %(default)s)")
    parser.add_argument("--stomp-iters", type=int, default=200,
                        help="STOMP max_iter (default %(default)s, ~2.5 s each)")
    parser.add_argument("--chomp-iters", type=int, default=1000,
                        help="CHOMP max_iter (default %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _cli()
    compare_coffe_table(stomp_iters=args.stomp_iters, chomp_iters=args.chomp_iters)
