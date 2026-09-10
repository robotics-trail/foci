"""Side-by-side comparison of FOCI, CHOMP and STOMP on a single scene.

Shared by the demos/compare_*.py scripts so the scenes cannot drift apart the
way the copy-pasted benchmark scripts did.

How the three are made comparable
---------------------------------
FOCI is given the TASK-space goal and its final configuration then becomes the
CONFIGURATION-space goal for CHOMP and STOMP, which is the only endpoint mode
those two implement.  Both are also warm-started from FOCI's RRT* path, so the
RRT* cost is charged once, to FOCI's row; their own initial-guess time is just
the resampling of that path.

`build` has to include the constructor: FOCI assembles its NLP and instantiates
IPOPT inside plan(), while the baselines prepare their CasADi callbacks and
matrices in __init__ and report a near-zero timings["build"].
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.benchmark.chomp_planner import CHOMPPlanner
from src.benchmark.stomp_planner import STOMPPlanner
from src.benchmark.utils import minimum_robot_environment_distance
from src.planning.planner import Planner
from src.planning.result import PlanningResult


# FOCI green, CHOMP blue, STOMP red.
PLANNER_COLORS: dict[str, tuple[int, int, int]] = {
    "FOCI":  (0, 200, 0),
    "CHOMP": (40, 90, 255),
    "STOMP": (235, 30, 30),
}


@dataclass
class PlannerRun:
    """One planner's result plus the timings needed for the metrics table."""

    name: str
    result: PlanningResult
    construct_time: float
    min_distance: float
    argmin: dict[str, Any] | None

    @property
    def initial_guess_time(self) -> float:
        return float(self.result.timings["initializer"])

    @property
    def build_time(self) -> float:
        return float(self.construct_time + self.result.timings["build"])

    @property
    def solve_time(self) -> float:
        return float(self.result.timings["solve"])


def _timed(factory: Callable[[], Any]) -> tuple[Any, float]:
    t0 = perf_counter()
    obj = factory()
    return obj, perf_counter() - t0


def run_comparison(
    *,
    robot,
    environment,
    joint_groups,
    start: np.ndarray,
    goal: np.ndarray,
    foci_options: dict[str, Any],
    stomp_options: dict[str, Any],
    chomp_options: dict[str, Any],
) -> list[PlannerRun]:
    """Run FOCI, CHOMP and STOMP on one scene. Returns one PlannerRun each."""
    common = dict(robot=robot, environment=environment, joint_groups=joint_groups)
    runs: list[PlannerRun] = []

    def measure(name, result, construct_time):
        d, argmin = minimum_robot_environment_distance(
            robot, environment, result.trajectory
        )
        run = PlannerRun(name, result, construct_time, float(d), argmin)
        runs.append(run)
        # Reported as each planner lands, not only in the final table: STOMP
        # can take tens of minutes, and without this there is no way to tell a
        # slow run from a hung one.
        md = result.metadata
        iters = f" iterations={md['iterations']}" if "iterations" in md else ""
        print(
            f"[{name}] done  initial_guess={run.initial_guess_time:.2f}s "
            f"build={run.build_time:.2f}s solve={run.solve_time:.2f}s "
            f"min_dist={run.min_distance:.4f}m success={result.success}{iters}",
            flush=True,
        )

    print(f"[FOCI] running ({environment.n_obstacles} environment Gaussians)...",
          flush=True)
    foci, t = _timed(lambda: Planner(**common, **foci_options))
    foci_result = foci.plan(start=start, goal=goal)
    measure("FOCI", foci_result, t)

    # Configuration-space goal for the baselines, and FOCI's RRT* path as their
    # warm start -- see the module docstring.
    theta_final = foci_result.trajectory[-1]
    warm_start = foci_result.initial_trajectory

    print(f"[CHOMP] running (max_iter={chomp_options.get('max_iter')})...", flush=True)
    chomp, t = _timed(lambda: CHOMPPlanner(**common, **chomp_options))
    measure("CHOMP",
            chomp.plan(start=start, goal=theta_final, initial_trajectory=warm_start),
            t)

    print(f"[STOMP] running (max_iter={stomp_options.get('max_iter')}, ~2.5 s/iter here)...", flush=True)
    stomp, t = _timed(lambda: STOMPPlanner(**common, **stomp_options))
    measure("STOMP",
            stomp.plan(start=start, goal=theta_final, initial_trajectory=warm_start),
            t)

    return runs


def print_metrics(
    scene: str,
    runs: list[PlannerRun],
    environment,
    n_robot_gaussians: int,
) -> None:
    """Print the per-planner metrics table."""
    n_env = int(environment.n_obstacles)

    print(f"\n=== {scene} " + "=" * max(0, 96 - len(scene)))
    print(
        f"{'planner':<8}{'Initial Guess [s]':>19}{'Build [s]':>12}"
        f"{'Solve [s]':>12}{'# Env Gaussians':>18}{'# Robot Gaussians':>20}"
        f"{'Min Safety Dist [m]':>22}"
    )
    print("-" * 111)
    for r in runs:
        print(
            f"{r.name:<8}{r.initial_guess_time:>19.3f}{r.build_time:>12.3f}"
            f"{r.solve_time:>12.3f}{n_env:>18d}{n_robot_gaussians:>20d}"
            f"{r.min_distance:>22.4f}"
        )
    print(
        "\nInitial Guess: RRT* for FOCI; CHOMP/STOMP resample FOCI's RRT* path,\n"
        "  so the search is charged once (to FOCI).\n"
        "Build: constructor + in-plan build, because the baselines do their\n"
        "  CasADi/matrix setup in __init__ and FOCI does it inside plan().\n"
        "Min Safety Dist: smallest centre-to-centre distance between any robot\n"
        "  Gaussian and any obstacle Gaussian (it ignores their extents)."
    )
    for r in runs:
        md = r.result.metadata
        extra = ""
        if "iterations" in md:
            # STOMP/CHOMP: say whether they stopped on their own or ran into
            # max_iter, since Solve Time is meaningless without it.
            extra = (f" iterations={md['iterations']}"
                     f" converged={md.get('converged', 'n/a')}")
        print(f"  {r.name:<6} success={str(r.result.success):<5} "
              f"samples={r.result.n_samples:<4} "
              f"closest at sample {r.argmin['sample'] if r.argmin else '-'}"
              f"{extra}")


def visualize_comparison(
    *,
    runs: list[PlannerRun],
    robot,
    goal: np.ndarray,
    splat_name: str,
    obstacle_means: np.ndarray,
    obstacle_covs: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    animate: bool = True,
    port: int | None = None,
):
    """Draw the scene, the robot and all three task-space paths.

    The animated robot follows FOCI's trajectory; every planner's path is drawn
    with its PLANNER_COLORS entry.
    """
    from src.visualization.visualizer import RobotVisualizer

    foci = next(r for r in runs if r.name == "FOCI")
    vis = RobotVisualizer(robot=robot, trajectory=foci.result.trajectory,
                          port=port)

    vis.visualize_goal(goal)
    vis.visualize_gaussian_splat(splat_name, obstacle_means, obstacle_covs,
                                 colors, opacities)
    vis.visualize_robot_gaussians()

    for run in runs:
        vis.visualize_initializer_path(
            run.result.trajectory,
            name=run.name,
            color=PLANNER_COLORS[run.name],
            line_width=5.0,
        )

    if animate:
        vis.visualize_trajectory(loop=True)

    return vis
