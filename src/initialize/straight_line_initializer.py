from __future__ import annotations

import numpy as np

from src.initialize.initializer import InitializerResult, PathInitializer

class StraightLineInitializer(PathInitializer):
    """
    Fallback initializer that holds the start configuration for all control points.

    A true straight-line interpolation in configuration space would require IK
    to resolve the task-space goal into a goal configuration, which is not
    available here. This constant warm-start is sufficient for the NLP to
    converge from.
    """

    def initialize(
        self,
        robot,
        environment,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ) -> InitializerResult:
        del robot, environment, goal

        start = np.asarray(start, dtype=float)
        control_points = np.tile(start, (num_control_points, 1))

        return InitializerResult(
            control_points=control_points,
            trajectory=control_points,
            success=True,
            timings={"initializer": 0.0},
            metadata={"type": "constant"},
        )