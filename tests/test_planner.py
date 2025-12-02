import pytest

import numpy as np
import casadi as cas

from src.planning.planner import Planner


@pytest.mark.order(0)
def test_planner_creation_and_plan():
    urdf_path = "urdfs/ur5.urdf"
    obstacle_means = np.array([[0.2, 0.0, 0.1]])
    obstacle_covs = np.array([np.eye(3) * 0.01])
    robot_cov = np.eye(3) * 0.01

    planner = Planner(
        urdf_file=urdf_path,
        root_link="base_link",
        tip_link="ee_link",
        obstacle_positions=obstacle_means,
        obstacle_covs=obstacle_covs,
        robot_cov=robot_cov,
        num_control_points=5,
        num_samples=10,
    )

    assert isinstance(planner.solver, cas.Function)
    assert len(planner.lbg) > 0
    assert len(planner.ubg) > 0

    theta_start = np.zeros(planner.n_joints)
    ee_goal = np.array([0.4, 0.1, 0.3])

    curve = planner.plan(theta_start, ee_goal)

    assert isinstance(curve, np.ndarray)
    assert curve.shape == (planner.num_samples, planner.n_joints)
    assert np.all(np.isfinite(curve))
