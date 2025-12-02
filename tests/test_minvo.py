import numpy as np
import casadi as cas

import pytest

from src.splines.minvo import minvo_hulls


@pytest.mark.order(5)
def test_minvo_hulls_shapes() -> None:
    cps = np.random.randn(10, 2)

    for d in [0, 1, 2, 3]:
        hulls = minvo_hulls(cps, derivative=d)

        if len(hulls) == 0:
            assert False, f"No hulls returned for derivative {d}"

        for h in hulls:
            if h.shape[1] != 2:
                assert False, f"Hull has wrong dimension: {h.shape}"


@pytest.mark.order(6)
def test_minvo_hulls_constant() -> None:
    cps = np.tile(np.array([2.0, 3.0]), (8, 1))

    for d in [0, 1, 2, 3]:
        hulls = minvo_hulls(cps, derivative=d)

        for h in hulls:
            assert np.allclose(
                h, h[0], atol=1e-6
            ), f"Constant control points produce non-constant hull for d={d}"


@pytest.mark.order(7)
def test_minvo_hulls_count() -> None:
    n = 10
    cps = np.random.randn(n, 2)

    for d in [0, 1, 2, 3]:
        hulls = minvo_hulls(cps, derivative=d)
        expected_count = n - 3 - d
        assert len(hulls) == expected_count, f"Wrong number of hulls for derivative {d}"


@pytest.mark.order(8)
def test_minvo_hulls_too_few_points() -> None:
    cps = np.random.randn(2, 2)
    for d in [0, 1, 2, 3]:
        hulls = minvo_hulls(cps, derivative=d)
        assert hulls == [] or len(hulls) == 0, "Hull should be empty for too few points"
