"""FOCI vs CHOMP vs STOMP on the Bonsai splat scene.

    python demos/compare_bonsai.py                    # metrics table only
    python demos/compare_bonsai.py --vis              # + viser on http://localhost:8080
    python demos/compare_bonsai.py --vis --port 9000  # pick the port
    python demos/compare_bonsai.py --stomp-iters 1000 # STOMP's own default

STOMP is gradient-free: every iteration scores 25-30 noisy trajectories
against every environment Gaussian, which costs ~2.5 s/iteration here.  Its
default is capped below the benchmark's 1000 so the demo finishes; the table
prints the iteration count and whether it converged on its own.
"""

import argparse
import sys

import numpy as np

_DEFAULT_PORT = 8080
from scipy.spatial.transform import Rotation as R

from src.benchmark.comparison import (
    print_metrics,
    run_comparison,
    visualize_comparison,
)
from src.environment.environment import GaussianEnvironment
from src.initialize.rrtstar_initializer import RRTStarInitializer
from src.planning.joints import JointGroups
from src.robots.manipulator import LinkGaussian, ManipulatorRobot
from src.utils.paths import data_path
from src.utils.ply import extract_splat_data_2


def compare_bonsai(stomp_iters: int = 200, chomp_iters: int = 1000):
    obstacle_means, obstacle_covs, colors, opacities = extract_splat_data_2(
        data_path("Bonsai.ply")
    )

    rotation = R.from_euler("x", -90, degrees=True).as_matrix()
    translation = np.array([0.0, 1.8, 1.0])
    scale_factor = 3.5

    obstacle_means = (obstacle_means * scale_factor) @ rotation.T + translation
    obstacle_covs = np.einsum(
        "ij,njk,lk->nil", rotation, obstacle_covs, rotation
    ) * scale_factor ** 2

    gaussian_specs = [
        LinkGaussian(0, 0.5, np.eye(3) * 0.1 ** 2),
        LinkGaussian(1, 0.5, np.eye(3) * 0.1 ** 2),
        LinkGaussian(2, 0.5, np.eye(3) * 0.1 ** 2),
        LinkGaussian(3, 0.5, np.eye(3) * 0.2 ** 2),
        LinkGaussian(4, 0.5, np.eye(3) * 0.2 ** 2),
        LinkGaussian(5, 0.5, np.eye(3) * 0.1 ** 2),
        LinkGaussian(6, 0.5, np.eye(3) * 0.1 ** 2),
        LinkGaussian(7, 0.5, np.eye(3) * 0.1 ** 2),
    ]

    theta_start = np.array([1.05, -0.23, -1.6, 1.21, -0.85, 0.02])
    goal = np.array([-1.5, 2.25, 0.5])

    robot = ManipulatorRobot(
        urdf_path="urdfs/ur5_extended.urdf",
        root_link="base_link",
        tip_link="ee_link",
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

    runs = run_comparison(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        start=theta_start,
        goal=goal,
        foci_options=dict(
            initializer=RRTStarInitializer(
                voxel_size=0.0001, goal_threshold=0.001, random_seed=42,
            ),
            num_control_points=12,
            num_samples=25,
            weights={"goal": 10.0, "obstacle": 1.0,
                     "jerk": 0.00049327, "virtual_jerk": 0.0049327},
            vmax=2.0,
            linear_solver="ma27",
        ),
        chomp_options=dict(
            num_waypoints=12, max_iter=chomp_iters, learning_rate=0.01,
            weights={"obstacle": 1.0, "smoothness": 1.0},
            convergence_tol=1e-3, total_time=2.0,
        ),
        stomp_options=dict(
            num_waypoints=12, n_samples=25, max_iter=stomp_iters, temperature=10.0,
            # obstacle = num_waypoints: _obstacle_cost_trajectory averages over
            # waypoints, so this keeps the obstacle/jerk/constraint balance the
            # benchmark was tuned with.
            weights={"obstacle": 12.0, "jerk": 1.0, "constraint": 1.0},
            noise_scale=0.1, convergence_tol=1e-3, seed=42, total_time=2.0,
        ),
    )

    print_metrics("Bonsai", runs, environment, len(gaussian_specs))

    if "--vis" in sys.argv:
        port = (int(sys.argv[sys.argv.index("--port") + 1])
                if "--port" in sys.argv else _DEFAULT_PORT)
        print(f"\nviser: http://localhost:{port}   "
              "(FOCI green, CHOMP blue, STOMP red)", flush=True)
        visualize_comparison(
            port=port,
            runs=runs, robot=robot, goal=goal, splat_name="Bonsai",
            obstacle_means=obstacle_means, obstacle_covs=obstacle_covs,
            colors=colors, opacities=opacities,
        )


def _cli():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vis", action="store_true",
                        help="serve the viser visualisation after the table")
    parser.add_argument("--port", type=int, default=8080,
                        help="viser port (default %(default)s)")
    parser.add_argument("--stomp-iters", type=int, default=200,
                        help="STOMP max_iter (default %(default)s, ~2.5 s each)")
    parser.add_argument("--chomp-iters", type=int, default=1000,
                        help="CHOMP max_iter (default %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _cli()
    compare_bonsai(stomp_iters=args.stomp_iters, chomp_iters=args.chomp_iters)
