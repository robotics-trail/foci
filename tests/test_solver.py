import pytest

import numpy as np
import casadi as cas

from src.core.robot_loader import ManipulatorRobotURDF
from src.optim.solver import create_solver


@pytest.mark.order(0)
def test_solver_creation() -> None:
    urdf_path = "urdfs/ur5.urdf"
    robot = ManipulatorRobotURDF(urdf_path, "base_link", "ee_link")

    obstacle_means = np.array([[0.2, 0.0, 0.1]])
    covs = np.array([np.eye(3) * 0.01])

    solver, lbg, ubg, _ = create_solver(
        robot=robot,
        num_control_points=5,
        obstacle_means=obstacle_means,
        covs_det=np.array([np.linalg.det(c) for c in covs]),
        covs_inv=np.array([np.linalg.inv(c) for c in covs]),
        num_samples=10,
        vmax=1.0,
        amax=1.0,
    )

    assert isinstance(solver, cas.Function)
    assert len(lbg) > 0
    assert len(ubg) > 0
