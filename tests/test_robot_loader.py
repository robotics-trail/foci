import numpy as np
import casadi as cas

import pytest

from src.core.robot_loader import ManipulatorRobotURDF, MobileManipulatorRobotURDF


@pytest.mark.order(0)
def test_robot_loader() -> None:
    ur5_path: str = "urdfs/ur5.urdf"

    ur5_loader = ManipulatorRobotURDF(
        ur5_path, root_link="base_link", tip_link="ee_link"
    )

    if ur5_loader.n_joints != 6:
        assert False, f"The number of joints is {ur5_loader.n_joints}, should be 6"

    q = cas.DM.zeros(ur5_loader.n_joints, 1)
    positions = ur5_loader.forward_kinematics(q)

    assert positions.dim() == "27x1"

    assert np.allclose(
        np.array([positions[0], positions[1], positions[2]]).flatten(),
        np.array([0.0, 0.0, 0.0]),
        atol=5e-2,
        rtol=1e-5,
    )

    assert np.allclose(
        np.array([positions[3], positions[4], positions[5]]).flatten(),
        np.array([0.0, 0.0, 0.1]),
        atol=5e-2,
        rtol=1e-5,
    )

    assert np.allclose(
        np.array([positions[-6], positions[-5], positions[-4]]).flatten(),
        np.array([0.0, 0.2, 1.2]),
        atol=5e-2,
        rtol=1e-5,
    )

    assert np.allclose(
        np.array([positions[-3], positions[-2], positions[-1]]).flatten(),
        np.array([0.0, 0.2, 1.205]),
        atol=5e-2,
        rtol=1e-5,
    )


@pytest.mark.order(1)
def test_robot_movile_loader() -> None:
    ur5_mobile_path: str = "urdfs/ur5_extended_move.urdf"

    ur5_loader = ManipulatorRobotURDF(
        ur5_mobile_path, root_link="world", tip_link="ee_link"
    )

    if ur5_loader.n_joints != 9:
        assert False, f"The number of joints is {ur5_loader.n_joints}, should be 9"

    q = cas.DM.zeros(ur5_loader.n_joints, 1)
    positions = ur5_loader.forward_kinematics(q)

    assert positions.dim() == "36x1"

    assert np.allclose(
        np.array([positions[0], positions[1], positions[2]]).flatten(),
        np.array([0.0, 0.0, 0.0]),
        atol=5e-2,
        rtol=1e-5,
    )


@pytest.mark.order(2)
def test_robot_get_joint_limits() -> None:
    ur5_mobile_path: str = "urdfs/ur5_extended_move.urdf"

    ur5_loader = ManipulatorRobotURDF(
        ur5_mobile_path, root_link="world", tip_link="ee_link"
    )

    n_joint_limits = ur5_loader.get_joint_limits()
    if len(n_joint_limits) != 9:
        assert False, f"The number of limit joints is {n_joint_limits}, should be 9"

    assert np.allclose(
        np.array(n_joint_limits[0]), np.array([-10.0, 10.0]), atol=5e-2, rtol=1e-5
    )

    assert np.allclose(
        np.array(n_joint_limits[-1]), np.array([-np.pi, np.pi]), atol=5e-2, rtol=1e-5
    )
