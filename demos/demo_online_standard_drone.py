"""
Demo: a non-spherical drone Gaussian rotating with the drone during planning.

StandardDroneRobot's single collision Gaussian is a flat disc, elongated
along the body's LOCAL Y axis (its rotor-arm span) and thin along local Z.
Flying level, this disc has a tiny vertical extent. Any lateral acceleration
banks the body, which swings that long axis up out of the horizontal plane
and into Z -- the drone's effective vertical footprint grows exactly when it
accelerates sideways.

The scene makes that sideways acceleration unavoidable: start and goal are
offset in y, so the drone has to drift laterally across the whole flight,
and a low ceiling slab sits right where that drift is fastest. Whether the
resulting tilt is accounted for or not changes what the optimiser considers
safe there -- that's wired through collision_covariances_online into
src/optimization/problem_online.py's obstacle cost (see
build_problem/_build_collision_cov_map). A MobileObstacle further along the
corridor exercises the dynamic-environment machinery from demo_online.py
without being the thing the comparison hinges on.

Both DroneRobot (omnidirectional, never tilts) and StandardDroneRobot
(tilt-aware) are solved on the IDENTICAL scene so they can be compared
directly -- see the printed realised clearance under the ceiling for each.

Run with --vis to open the viser visualization of the StandardDroneRobot
solution: the ellipsoid should visibly tilt as the trajectory drifts
sideways, most while passing under the ceiling.
"""

import argparse

import numpy as np

from src.robots.drone import DroneRobot, StandardDroneRobot, DroneGaussian, finite_difference_acceleration
from src.environment.environment import GaussianEnvironmentOnline
from src.environment.obstacle import StaticObstacle, MobileObstacle
from src.planning.planner_online import OnlinePlanner
from src.planning.joints import JointGroups
from src.initialize.straight_line import StraightLineInitializer
from src.visualization.visualizer import RobotVisualizerOnline


NUM_SAMPLES = 24
CEILING_MEAN = np.array([2.5, 0.0, 1.45])
CEILING_COV = np.diag([1.0**2, 2.5**2, 0.12**2])


def _build_environment() -> GaussianEnvironmentOnline:
    ceiling = StaticObstacle(CEILING_MEAN, CEILING_COV)

    # A second, independent obstacle drifting slowly well downstream of the
    # ceiling, exercising GaussianEnvironmentOnline's per-sample mobile
    # obstacle query without competing with the ceiling for the same dodge.
    s = np.linspace(0.0, 1.0, NUM_SAMPLES)
    sweep = np.stack([np.zeros_like(s), -1.5 + 1.0 * s, np.zeros_like(s)], axis=1)
    mobile = MobileObstacle(np.array([5.0, 1.5, 1.0]), np.diag([0.2**2, 0.2**2, 0.2**2]), sweep)

    return GaussianEnvironmentOnline(obstacles=[ceiling, mobile])


def _build_robot(standard: bool):
    # Three DISTINCT semi-axes on purpose: 0.6 along local y (rotor-arm
    # span), 0.4 along local x (nose-tail), 0.2 along local z (top-bottom).
    # If any two matched, the ellipsoid would be a body of revolution about
    # the axis they share, and rotating about that shared axis wouldn't
    # change its rendered shape at all -- e.g. x == z would make pitch
    # (forward/backward acceleration, a rotation about y) invisible, only
    # roll (lateral acceleration) would show. With all three different,
    # both roll and pitch visibly move the ellipsoid.
    specs = [DroneGaussian(np.array([0.0, 0.0, 0.0]), np.diag([0.4**2, 0.6**2, 0.2**2]))]
    cls = StandardDroneRobot if standard else DroneRobot
    return cls(urdf_path="urdfs/drone_example.urdf", gaussian_specs=specs)


def _tilt_diagnostics(robot, trajectory: np.ndarray, duration: float) -> tuple[float, float]:
    """
    Returns (min ceiling clearance, max tilt angle in degrees) for this
    trajectory.

    qdd is a finite-difference estimate over the SOLVED trajectory, scaled
    by the planner's own real duration (NOT the unit index-spacing
    finite_difference_acceleration defaults to) -- using the wrong dt here
    would under- or over-state the acceleration, and with it the tilt, by
    whatever factor dt^2 is off by.

    The clearance check (ceiling 1-sigma surface vs. the robot Gaussian's
    own extent along the same axis, restricted to samples actually
    underneath the slab) is deliberately a simple 1-D check along z, not
    the full ellipsoid bisection in benchmark/utils.py -- so the effect this
    demo is about, the long axis swinging into z, stays legible as one
    number.
    """
    num_samples = trajectory.shape[0]
    dt = duration / max(num_samples - 1, 1)

    qdd = (
        finite_difference_acceleration(trajectory, dt=dt)
        if robot.requires_acceleration
        else np.zeros_like(trajectory)
    )

    ceiling_half_extent_z = float(np.sqrt(CEILING_COV[2, 2]))
    min_gap = np.inf
    max_tilt_deg = 0.0

    for t, q in enumerate(trajectory):
        cov = np.asarray(robot.collision_covariances_online(q, qdd[t]), dtype=float).reshape(3, 3)
        robot_half_extent_z = float(np.sqrt(cov[2, 2]))

        eigvals, eigvecs = np.linalg.eigh(cov)
        long_axis = eigvecs[:, np.argmax(eigvals)]
        tilt_deg = float(np.degrees(np.arcsin(np.clip(abs(long_axis[2]), 0.0, 1.0))))
        max_tilt_deg = max(max_tilt_deg, tilt_deg)

        point = np.asarray(robot.collision_points(q, qdd[t]), dtype=float).reshape(3)
        horizontal_dist = np.linalg.norm(point[:2] - CEILING_MEAN[:2])

        if horizontal_dist < 1.2:
            gap = abs(point[2] - CEILING_MEAN[2]) - ceiling_half_extent_z - robot_half_extent_z
            min_gap = min(min_gap, gap)

    return min_gap, max_tilt_deg


def _build_synthetic_accel_brake_trajectory(
    num_samples: int, duration: float, peak_accel: float, altitude: float = 1.0,
) -> np.ndarray:
    """
    A hand-built (not solved) trajectory, straight along x: constant
    acceleration +peak_accel for the first half of the flight, constant
    deceleration -peak_accel for the second half (a triangular velocity
    profile). No jerk-minimisation, no obstacles pulling it elsewhere --
    the acceleration is exactly peak_accel throughout each half, so the
    tilt it produces is as large and as unambiguous as it gets, instead of
    the mild, ramping accelerations an optimised, smooth trajectory settles
    for.
    """
    t = np.linspace(0.0, duration, num_samples)
    half = duration / 2.0
    accelerating = t <= half

    x = np.where(
        accelerating,
        0.5 * peak_accel * t**2,
        0.5 * peak_accel * half**2 + peak_accel * half * (t - half) - 0.5 * peak_accel * (t - half) ** 2,
    )

    return np.stack([x, np.zeros_like(t), np.full_like(t, altitude), np.zeros_like(t)], axis=1)


def run_synthetic(show_vis: bool, peak_accel: float = 6.0, duration: float = 6.0, num_samples: int = 60) -> None:
    """
    Skip planning entirely: feed StandardDroneRobot a hand-built
    accelerate-then-brake trajectory directly, so the tilt this is all
    about is as large and unmissable as possible, rather than whatever
    mild amount an optimiser's jerk-minimising solution happens to need.
    """
    robot = _build_robot(standard=True)
    trajectory = _build_synthetic_accel_brake_trajectory(num_samples, duration, peak_accel)

    dt = duration / (num_samples - 1)
    qdd = finite_difference_acceleration(trajectory, dt=dt)
    expected_tilt_deg = np.degrees(np.arctan(peak_accel / 9.81))

    # This motion is pure pitch (rotation about local y, no lateral
    # acceleration), so the LONG axis (local y, the arm span) is the pitch
    # rotation axis itself and stays fixed -- it is the local-x ("nose")
    # semi-axis that swings into z. Track whichever eigenvector's eigenvalue
    # is closest to local x's (0.4**2), not the largest one.
    local_semi_axes_sq = np.array([0.4**2, 0.6**2, 0.2**2])
    nose_variance = local_semi_axes_sq[0]

    max_tilt_deg = 0.0
    for t, q in enumerate(trajectory):
        cov = np.asarray(robot.collision_covariances_online(q, qdd[t]), dtype=float).reshape(3, 3)
        eigvals, eigvecs = np.linalg.eigh(cov)
        nose_axis = eigvecs[:, int(np.argmin(np.abs(eigvals - nose_variance)))]
        tilt_deg = float(np.degrees(np.arcsin(np.clip(abs(nose_axis[2]), 0.0, 1.0))))
        max_tilt_deg = max(max_tilt_deg, tilt_deg)

    print(f"\n--- Synthetic accelerate-then-brake trajectory (peak accel = {peak_accel} m/s^2) ---")
    print(f"Expected pitch from a = {peak_accel} m/s^2: {expected_tilt_deg:.2f} deg (arctan(a/g))")
    print(f"Measured peak tilt (nose axis vs. z, from collision_covariances_online): {max_tilt_deg:.2f} deg")

    if show_vis:
        vis = RobotVisualizerOnline(robot=robot, trajectory=trajectory, duration=duration)
        vis.visualize_robot_gaussians()
        vis.visualize_path()
        vis.visualize_trajectory(loop=True)


def run(standard: bool, show_vis: bool):
    environment = _build_environment()
    robot = _build_robot(standard)

    joint_groups = JointGroups(
        virtual_indices=[0, 1, 2],
        virtual_wmax=4.0,
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
            "obstacle": 0.3,
            "jerk": 0.01,
            "virtual_jerk": 0.01,
        },
        vmax=2.0,
        linear_solver="mumps",
    )

    # Offset in y forces a lateral drift across the whole flight; the
    # ceiling at x = 2.5 sits right where that drift -- and the resulting
    # bank -- is largest.
    start = np.array([-1.0, -2.0, 1.0, 0.0])
    goal = np.array([6.0, 2.0, 1.0])

    result = planner.plan(start=start, goal=goal)
    duration = max(float(np.linalg.norm(goal - start[:3])) / planner.vmax, 1e-3)

    label = "StandardDroneRobot (tilt-aware)" if standard else "DroneRobot (omnidirectional, never tilts)"
    print(f"\n--- {label} ---")
    print(f"Success: {result.success}  ({result.solver_stats.get('return_status')})")
    print(f"Solve time: {result.timings['solve']:.3f} s")

    clearance, max_tilt_deg = _tilt_diagnostics(robot, result.trajectory, duration)
    print(f"Peak body tilt during the flight: {max_tilt_deg:.2f} deg")
    print(f"Realised clearance under the ceiling (negative = clipped): {clearance:+.4f} m")

    if show_vis:
        vis = RobotVisualizerOnline(robot=robot, trajectory=result.trajectory, duration=duration)
        vis.visualize_goal(goal)
        vis.visualize_obstacles_online(environment)
        vis.visualize_robot_gaussians()
        vis.visualize_path()
        vis.visualize_initializer_path(result.initial_trajectory, color=(0, 0, 255), name="Initializer")
        vis.visualize_trajectory(loop=True)

    return result, clearance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", action="store_true", help="open the viser visualization")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="skip planning; directly visualize a hand-built accelerate-then-brake trajectory",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic(show_vis=args.vis)
    else:
        print("Solving the identical scene for both robot models...")
        _, clearance_standard = run(standard=True, show_vis=args.vis)
        _, clearance_omni = run(standard=False, show_vis=False)

        print("\n--- Comparison ---")
        print(f"StandardDroneRobot clearance: {clearance_standard:+.4f} m")
        print(f"DroneRobot (omni)  clearance: {clearance_omni:+.4f} m")
        print(
            "Same obstacles, same non-spherical Gaussian, same weights -- the "
            "only difference is whether the optimiser's own model of the robot "
            "lets the Gaussian's orientation respond to commanded acceleration. "
            "The resulting plans, and the realised clearance under the ceiling, "
            "differ because of that alone."
        )
