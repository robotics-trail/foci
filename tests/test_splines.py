import numpy as np
import casadi as cas
import pytest

from src.splines.bspline import BSpline


@pytest.mark.order(0)
def test_bspline_numpy_casadi_compatibility():
    # Test with numpy array
    control_points_np = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
        ]
    )

    spline_np = BSpline(control_points_np)
    samples_np = spline_np.spline_eval(10)  # 10 points

    assert isinstance(samples_np, np.ndarray), "NumPy version should return numpy array"
    assert samples_np.shape == (
        10,
        2,
    ), f"Expected shape (10, 2), got {samples_np.shape}"

    # Test with casadi MX
    # control_points_mx = cas.MX(
    #     [
    #         [0.0, 0.0],
    #         [1.0, 1.0],
    #         [2.0, 2.0],
    #         [3.0, 3.0],
    #         [4.0, 4.0],
    #         [5.0, 5.0],
    #         [6.0, 6.0],
    #     ]
    # )

    # spline_mx = BSpline(control_points_mx)
    # samples_mx = spline_mx.spline_eval(10)  # 10 points

    # assert isinstance(samples_mx, cas.MX), "CasADi version should return casadi MX"
    # assert samples_mx.shape == (
    #     10,
    #     2,
    # ), f"Expected shape (10, 2), got {samples_mx.shape}"


@pytest.mark.order(1)
def test_bspline_derivatives():
    """Test that derivatives work with both types"""

    # NumPy test
    control_points_np = np.random.rand(7, 3)
    spline_np = BSpline(control_points_np)

    samples_np_der0 = spline_np.spline_eval(5, der=0)
    samples_np_der1 = spline_np.spline_eval(5, der=1)

    assert samples_np_der0.shape == (5, 3)
    assert samples_np_der1.shape == (5, 3)

    # CasADi test
    control_points_mx = cas.MX(control_points_np)
    spline_mx = BSpline(control_points_mx)

    samples_mx_der0 = spline_mx.spline_eval(5, der=0)
    samples_mx_der1 = spline_mx.spline_eval(5, der=1)

    assert samples_mx_der0.shape == (5, 3)
    assert samples_mx_der1.shape == (5, 3)
