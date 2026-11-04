import sys
import select
import time
import os

import numpy as np

from src.utils.ply import extract_splat_data
from src.robots.drone import StandardDroneRobot, DroneGaussian
from src.environment.environment import GaussianEnvironmentOnline
from src.environment.obstacle import StaticObstacle, MobileObstacle
from src.planning.planner_online import OnlinePlanner
from src.planning.joints import JointGroups
from src.initialize.straight_line import StraightLineInitializer
from src.visualization.visualizer import RobotVisualizerOnline


# Both source clouds are dense captured point clouds (one Gaussian per point);
# cap each at this many Gaussians so viser can render the scene smoothly.
TARGET_MAX_POINTS = 80_000

N_BARRELS = 5
BARREL_SIZE_FRACTION = 0.10   # barrel size relative to the rock's size
ROCK_SCALE = 2.2              # enlarge the rock beyond its raw captured size
ORBIT_CLEARANCE = 0.4         # extra radius added beyond the rock's footprint
ORBIT_SWEEP = np.radians(70)  # how far each barrel travels along its epitrochoid during the flight

# Per-barrel variety, indexed by i = 0..N_BARRELS-1, so flying straight over
# everything at a single altitude is no longer an option: each barrel sits at
# its own height and follows its own epitrochoid (its own orbit radius, lobe
# count, loop sharpness and sweep direction).
BARREL_HEIGHT_FRACTIONS = np.linspace(0.15, 1.1, N_BARRELS)   # fraction of the rock's height
BARREL_RADIUS_FACTORS = np.linspace(0.8, 1.3, N_BARRELS)      # fraction of the base orbit radius
BARREL_LOBES = 3 + np.arange(N_BARRELS)                       # R/r ratio -> number of lobes
BARREL_LOOPINESS = np.linspace(0.3, 0.7, N_BARRELS)           # d/r ratio -> loop sharpness
BARREL_SWEEP_SIGNS = np.where(np.arange(N_BARRELS) % 2 == 0, 1.0, -1.0)

NUM_SAMPLES = 24

# extract_splat_data has no real per-point scale (this .ply has no scale_*
# columns), so it fills covariances with sigma = 1 m identity -- on a multi-
# metre object that is one giant blob per point, totally overlapping. Shrink
# each point to a plausible dot radius before display. It is kept fixed
# (not tied to the barrel's own size) so the dots don't shrink below a
# visible size once the barrel itself is scaled down.
POINT_SIGMA = 0.01

# A point cloud's variance underestimates the true half-extent of a filled
# solid (e.g. ~1/3 of it, for a uniform-filled box), so inflate the fitted
# covariance when approximating a splat as a single Gaussian obstacle for
# planning -- otherwise the optimizer would see a "safe" region that is
# actually still inside the object.
OBSTACLE_INFLATION = 3.0


def _splat_size(means: np.ndarray) -> float:
    """Bounding-box diagonal of a point cloud -- the "size" reported for a 3DGS."""
    extent = means.max(axis=0) - means.min(axis=0)
    return float(np.linalg.norm(extent))


def _random_subsample(
    means: np.ndarray,
    covariances: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    keep_fraction: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep a random keep_fraction of the Gaussians, without replacement."""
    n = means.shape[0]
    n_keep = max(1, int(round(n * keep_fraction)))
    idx = np.random.default_rng(seed).choice(n, size=n_keep, replace=False)
    return means[idx], covariances[idx], colors[idx], opacities[idx]


def _ground_and_center(means: np.ndarray) -> np.ndarray:
    """Translate a point cloud so it is centered on x/y and resting on z = 0."""
    centroid_xy = means[:, :2].mean(axis=0)
    min_z = means[:, 2].min()
    return means - np.array([centroid_xy[0], centroid_xy[1], min_z])


def _epitrochoid_point(theta: float, R: float, r: float, d: float) -> tuple[float, float]:
    """
    Epitrochoid: traced by a point at distance d from the center of a circle
    of radius r rolling, without slipping, on the outside of a fixed circle
    of radius R.
    """
    k = (R + r) / r
    x = (R + r) * np.cos(theta) - d * np.cos(k * theta)
    y = (R + r) * np.sin(theta) - d * np.sin(k * theta)
    return x, y


def _fit_obstacle_gaussian(means: np.ndarray, inflation: float = OBSTACLE_INFLATION) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a point cloud into a single Gaussian (mean + inflated diagonal
    covariance) usable as a planning obstacle -- the dense splat itself has
    far too many points to hand to the online planner one by one."""
    mean = means.mean(axis=0)
    cov = np.diag(means.var(axis=0)) * inflation
    return mean, cov


def barrel_rock_epitrochoid_demo():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rock_file = os.path.join(PROJECT_ROOT, "data", "Rock on forest path.ply")
    barrel_file = os.path.join(PROJECT_ROOT, "data", "Barrel.ply")

    rock_means, rock_covs, rock_colors, rock_opacities = extract_splat_data(rock_file)
    barrel_means, barrel_covs, barrel_colors, barrel_opacities = extract_splat_data(barrel_file)

    rock_size = _splat_size(rock_means)
    barrel_size = _splat_size(barrel_means)

    print("--- 3DGS analysis (raw, before filtering/resizing) ---")
    print(f"Rock:   {len(rock_means):>8} gaussians, bounding-box size (diagonal) = {rock_size:.3f} m")
    print(f"Barrel: {len(barrel_means):>8} gaussians, bounding-box size (diagonal) = {barrel_size:.3f} m")

    # if len(rock_means) > TARGET_MAX_POINTS:
    #     keep_fraction = TARGET_MAX_POINTS / len(rock_means)
    #     rock_means, rock_covs, rock_colors, rock_opacities = _random_subsample(
    #         rock_means, rock_covs, rock_colors, rock_opacities, keep_fraction
    #     )
    #     print(f"\nRock is too dense -> random filter kept {keep_fraction:.1%} "
    #           f"({len(rock_means)} gaussians).")

    # if len(barrel_means) > TARGET_MAX_POINTS:
    #     keep_fraction = TARGET_MAX_POINTS / len(barrel_means)
    #     barrel_means, barrel_covs, barrel_colors, barrel_opacities = _random_subsample(
    #         barrel_means, barrel_covs, barrel_colors, barrel_opacities, keep_fraction
    #     )
    #     print(f"Barrel is too dense -> random filter kept {keep_fraction:.1%} "
    #           f"({len(barrel_means)} gaussians).")

    rock_size *= ROCK_SCALE
    print(f"\nEnlarging rock by {ROCK_SCALE:.3f}x -> new size = {rock_size:.3f} m.")

    barrel_scale = (BARREL_SIZE_FRACTION * rock_size) / barrel_size
    print(f"Scaling barrel by {barrel_scale:.3f}x -> new size = "
          f"{barrel_scale * barrel_size:.3f} m ({BARREL_SIZE_FRACTION:.0%} of the rock).")

    rock_means = _ground_and_center(rock_means) * ROCK_SCALE
    rock_covs = np.tile(np.eye(3) * POINT_SIGMA**2, (len(rock_means), 1, 1))

    barrel_means = _ground_and_center(barrel_means) * barrel_scale
    barrel_covs = np.tile(np.eye(3) * POINT_SIGMA**2, (len(barrel_means), 1, 1))

    rock_half_extent_xy = float(
        (rock_means[:, :2].max(axis=0) - rock_means[:, :2].min(axis=0)).max() / 2.0
    )
    rock_height = float(rock_means[:, 2].max())

    base_orbit_R = rock_half_extent_xy + ORBIT_CLEARANCE

    # --- Simplified Gaussian obstacles for planning (the dense splats above
    # are for display only -- far too many points to hand to the online
    # planner directly). ---

    rock_obstacle_mean, rock_obstacle_cov = _fit_obstacle_gaussian(rock_means)
    rock_obstacle = StaticObstacle(rock_obstacle_mean, rock_obstacle_cov)

    barrel_obstacle_mean_local, barrel_obstacle_cov = _fit_obstacle_gaussian(barrel_means)

    barrel_base_xyz = []
    barrel_trajectories = []
    barrel_obstacles = []
    max_orbit_reach = 0.0

    for i in range(N_BARRELS):
        orbit_R = base_orbit_R * float(BARREL_RADIUS_FACTORS[i])
        orbit_r = orbit_R / float(BARREL_LOBES[i])
        orbit_d = orbit_r * float(BARREL_LOOPINESS[i])
        height = float(BARREL_HEIGHT_FRACTIONS[i]) * rock_height

        theta0 = i * 2.0 * np.pi / N_BARRELS
        sweep = float(BARREL_SWEEP_SIGNS[i]) * ORBIT_SWEEP
        thetas = theta0 + np.linspace(0.0, sweep, NUM_SAMPLES)
        xy = np.array([_epitrochoid_point(theta, orbit_R, orbit_r, orbit_d) for theta in thetas])

        base_xyz = np.array([xy[0, 0], xy[0, 1], height])
        trajectory = np.column_stack([xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1], np.zeros(NUM_SAMPLES)])

        barrel_base_xyz.append(base_xyz)
        barrel_trajectories.append(trajectory)
        barrel_obstacles.append(
            MobileObstacle(barrel_obstacle_mean_local + base_xyz, barrel_obstacle_cov, trajectory)
        )
        max_orbit_reach = max(max_orbit_reach, orbit_R + orbit_r + orbit_d)

    environment = GaussianEnvironmentOnline(obstacles=[rock_obstacle] + barrel_obstacles)

    gaussian_specs = [DroneGaussian(np.array([0.0, 0.0, 0.0]), np.diag([0.25**2, 0.35**2, 0.12**2]))]
    robot = StandardDroneRobot(urdf_path="urdfs/drone_example.urdf", gaussian_specs=gaussian_specs)

    joint_groups = JointGroups(
        virtual_indices=[0, 1, 2],
        virtual_wmax=8.0,
        virtual_amax=6.0,
        real_wmax=4.0,
        real_amax=6.0,
    )

    planner = OnlinePlanner(
        robot=robot,
        environment=environment,
        joint_groups=joint_groups,
        initializer=StraightLineInitializer(),
        num_control_points=12,
        num_samples=NUM_SAMPLES,
        weights={
            "goal": 10.0,
            "obstacle": 2.0,
            "jerk": 0.01,
            "virtual_jerk": 0.01,
        },
        vmax=2.0,
        linear_solver="mumps",
    )

    # Start and goal sit on opposite sides of the rock, beyond the widest
    # barrel orbit, at the obstacle course's median height -- cruising
    # straight across at that altitude runs into the rock and several
    # barrels, so the drone actually has to weave between them.
    flight_radius = max_orbit_reach + 1.0
    flight_height = 0.5 * rock_height
    start = np.array([-flight_radius, 0.0, flight_height, 0.0])
    goal = np.array([flight_radius, 0.0, flight_height])

    result = planner.plan(start=start, goal=goal)
    duration = max(float(np.linalg.norm(goal - start[:3])) / planner.vmax, 1e-3)

    print(f"\n--- StandardDroneRobot through the rock + {N_BARRELS}-barrel epitrochoid ring ---")
    print(f"Success: {result.success}  ({result.solver_stats.get('return_status')})")
    print(f"Solve time: {result.timings['solve']:.3f} s")

    vis = RobotVisualizerOnline(robot=robot, trajectory=result.trajectory, duration=duration)
    vis.visualize_goal(goal)
    vis.visualize_robot_gaussians()
    vis.visualize_path()

    vis.server.scene.add_gaussian_splats(
        name="Rock",
        centers=rock_means,
        covariances=rock_covs,
        rgbs=rock_colors,
        opacities=rock_opacities,
    )

    barrel_handles = [
        vis.server.scene.add_gaussian_splats(
            name=f"Barrel {i}",
            centers=barrel_means,
            covariances=barrel_covs,
            rgbs=barrel_colors,
            opacities=barrel_opacities,
            position=barrel_base_xyz[i],
        )
        for i in range(N_BARRELS)
    ]

    print("\nVisualizing scene. Press 'q' + Enter to quit.")
    sample_idx = 0

    while True:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == "q":
                print("Stopping visualization.")
                break

        q = result.trajectory[sample_idx]
        vis._update_robot(q, sample_idx)
        vis._update_robot_gaussians(sample_idx)
        vis._update_camera(sample_idx)

        for i, handle in enumerate(barrel_handles):
            handle.position = barrel_base_xyz[i] + barrel_trajectories[i][sample_idx]

        time.sleep(0.1)

        sample_idx = (sample_idx + 1) % NUM_SAMPLES


if __name__ == "__main__":
    barrel_rock_epitrochoid_demo()
